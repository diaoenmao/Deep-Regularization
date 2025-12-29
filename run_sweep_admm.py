"""Minimal ADMM sweep script writing results to CSV.

Usage:
    python run_sweep_admm.py --epochs 1 --train-steps 120 --val-size 1000 \
        --c-values 0.5 1.0 --lr-values 0.001 0.0005 --outfile results/sweeps/admm_sweep.csv

Defaults are intentionally small for quick iteration.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from network.cnn3 import CNN
from optimizer.ADMM_global import ADMM_Adam_global
from Score.wanda_score import WANDA_ScoreCalculator
from Score.get_grad import GradientCollector
from Score.score_choos import choose_score


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def run_once(device, c_val: float, lr: float, train_steps: int, epochs: int, val_size: int):
    set_seed(42)
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=os.path.join("data", "MNIST"), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=os.path.join("data", "MNIST"), train=False, download=True, transform=transform)

    # small subsets for speed
    train_idx = list(range(0, 512))
    val_idx = list(range(0, min(val_size, len(test_ds))))
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(test_ds, val_idx), batch_size=128, shuffle=False)

    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = ADMM_Adam_global(
        params,
        lr=lr,
        N=len(train_ds),
        C=c_val,
        vk=zeros_like,
        wk=zeros_like,
        yk=zeros_like,
        zk=zeros_like,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for epoch in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            # second order (Wanda) scores each step
            score_dict = choose_score(wanda_calc, grad_collector, "second order")
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(new)

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    val_acc = correct / total if total > 0 else 0.0
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--lr-values", nargs="+", type=float, default=[1e-3])
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--outfile", type=str, default="results/sweeps/admm_sweep.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    rows = []
    for c_val in args.c_values:
        for lr in args.lr_values:
            val_acc, sparsity, steps = run_once(device, c_val, lr, args.train_steps, args.epochs, args.val_size)
            rows.append({
                "C": c_val,
                "lr": lr,
                "train_steps": steps,
                "val_size": args.val_size,
                "val_acc": val_acc,
                "sparsity": sparsity,
            })
            print(f"C={c_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")

    with open(args.outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["C", "lr", "train_steps", "val_size", "val_acc", "sparsity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved sweep results to {args.outfile}")


if __name__ == "__main__":
    main()
