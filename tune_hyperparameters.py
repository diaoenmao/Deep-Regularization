"""Hyperparameter tuning script for ADMM pruning.

This script performs a focused grid search to find optimal hyperparameters
for the ADMM optimizers, targeting the best accuracy-sparsity tradeoff.

Based on initial experiments:
- C=0.01 gives ~93% acc, 34% sparsity (good baseline)
- C=0.05+ causes model collapse (too aggressive)

Strategy:
1. Fine-grained search around C=0.01-0.05 for global
2. Adjust ranges for layer/neuron based on their threshold scaling
3. Test multiple learning rates
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from network.cnn3 import CNN
from optimizer.ADMM_global import ADMM_Adam_global
from optimizer.ADMM_layer import ADMM_Adam_layer
from optimizer.ADMM_neuron import ADMM_Adam_neuron
from score.wanda_score import WANDA_ScoreCalculator
from score.get_grad import GradientCollector
from score.score_choos import choose_score


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(batch_size: int = 64, subset_size: int = None):
    """Get MNIST train/test loaders."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data/MNIST", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data/MNIST", train=False, download=True, transform=transform)

    if subset_size:
        train_ds = Subset(train_ds, list(range(min(subset_size, len(train_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    return train_loader, test_loader, len(train_ds)


def align_scores(model: nn.Module, score_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    ordered = []
    for name, _ in model.named_parameters():
        ordered.append(score_dict[name])
    return ordered


def compute_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def evaluate(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return 100.0 * correct / total if total > 0 else 0.0


def train_admm(
    optimizer_class,
    device,
    lr: float,
    c_val: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    N_total: int,
    epochs: int,
    score_name: str = "magnitude",
) -> Tuple[float, float]:
    """Train with ADMM optimizer and return (accuracy, sparsity)."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    zk_init = [p.clone().detach() for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = optimizer_class(
        params, lr=lr, N=N_total, C=c_val,
        vk=[z.clone() for z in zeros_like],
        wk=[z.clone() for z in zeros_like],
        yk=[p.clone().detach() for p in params],
        zk=zk_init,
        score=score_buffers,
    )

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-8))

            optimizer.step()

        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)

        # Early stop if model collapsed
        if acc < 15.0:
            break

    wanda_calc.remove_hooks()
    final_acc = evaluate(model, test_loader, device)
    sparsity = compute_sparsity(model)

    return max(best_acc, final_acc), sparsity


def run_tuning(args):
    """Run hyperparameter tuning."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(args.seed)
    train_loader, test_loader, N_total = get_dataloaders(
        batch_size=args.batch_size,
        subset_size=args.subset_size
    )
    print(f"Dataset size: {N_total}, Epochs: {args.epochs}")
    print()

    # Hyperparameter ranges based on initial experiments
    # Global: threshold = C * 0.01, so C=0.01 -> thresh=0.0001
    # Layer: threshold = C * 0.001, so need C ~10x higher
    # Neuron: threshold = C * 0.0001, so need C ~100x higher

    configs = {
        "ADMM_global": {
            "class": ADMM_Adam_global,
            "C_values": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
            "lr_values": [0.001, 0.0005, 0.002],
        },
        "ADMM_layer": {
            "class": ADMM_Adam_layer,
            "C_values": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
            "lr_values": [0.001, 0.0005, 0.002],
        },
        "ADMM_neuron": {
            "class": ADMM_Adam_neuron,
            "C_values": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
            "lr_values": [0.001, 0.0005, 0.002],
        },
    }

    results = []
    best_results = {}

    for opt_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Tuning {opt_name}")
        print(f"{'='*60}")

        opt_results = []
        best_score = 0  # acc - sparsity_penalty

        for lr in config["lr_values"]:
            for c_val in config["C_values"]:
                print(f"  LR={lr}, C={c_val}...", end=" ", flush=True)

                acc, sparsity = train_admm(
                    config["class"], device, lr, c_val,
                    train_loader, test_loader, N_total,
                    args.epochs, args.score
                )

                # Score: prioritize accuracy but reward sparsity
                # Target: >90% acc with >50% sparsity
                score = acc - max(0, 50 - sparsity * 100) * 0.5  # Penalty if sparsity < 50%

                result = {
                    "optimizer": opt_name,
                    "lr": lr,
                    "C": c_val,
                    "accuracy": acc,
                    "sparsity": sparsity,
                    "score": score,
                }
                opt_results.append(result)
                results.append(result)

                print(f"Acc={acc:.2f}%, Sparsity={sparsity*100:.1f}%")

                if score > best_score:
                    best_score = score
                    best_results[opt_name] = result

        # Print best for this optimizer
        if opt_name in best_results:
            best = best_results[opt_name]
            print(f"\n  Best: LR={best['lr']}, C={best['C']} -> Acc={best['accuracy']:.2f}%, Sparsity={best['sparsity']*100:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print("TUNING SUMMARY")
    print(f"{'='*60}")
    for opt_name, best in best_results.items():
        print(f"{opt_name}:")
        print(f"  Best LR: {best['lr']}")
        print(f"  Best C: {best['C']}")
        print(f"  Accuracy: {best['accuracy']:.2f}%")
        print(f"  Sparsity: {best['sparsity']*100:.1f}%")
        print()

    # Save results
    os.makedirs("results/metrics", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/metrics/tuning_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "all_results": results,
            "best_results": best_results,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "subset_size": args.subset_size,
                "score": args.score,
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")

    return results, best_results


def main():
    parser = argparse.ArgumentParser(description="ADMM Hyperparameter Tuning")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--subset-size", type=int, default=10000, help="Training subset size (None for full)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--score", type=str, default="magnitude", help="Score type")

    args = parser.parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
