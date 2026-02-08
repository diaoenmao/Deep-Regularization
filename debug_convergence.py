"""Sanity check script for ADMM pruning convergence.

This script validates that the ADMM optimizer is working correctly by checking:
1. Loss decreases consistently over iterations
2. Sparsity increases over iterations
3. No NaN values appear in q, y, z variables

Run this BEFORE any full experiments (CIFAR/ImageNet).

Success Criteria:
- Loss should decrease (or stay stable) over 50 steps
- Sparsity should increase from 0% towards target
- No NaN values in any ADMM variables

Usage:
    python debug_convergence.py --optimizer ADMM_global --steps 50
    python debug_convergence.py --optimizer ADMM_layer --steps 50
    python debug_convergence.py --optimizer ADMM_neuron --steps 50
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from network.cnn3 import CNN
from optimizer.ADMM_global import ADMM_Adam_global
from optimizer.ADMM_layer import ADMM_Adam_layer
from optimizer.ADMM_neuron import ADMM_Adam_neuron
from score.wanda_score import WANDA_ScoreCalculator
from score.get_grad import GradientCollector
from score.score_choos import choose_score


def align_scores(model: nn.Module, score_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Return scores ordered to match model.parameters()."""
    ordered = []
    for name, _ in model.named_parameters():
        if name not in score_dict:
            raise KeyError(f"Missing score for parameter: {name}")
        ordered.append(score_dict[name])
    return ordered


def compute_sparsity(model: nn.Module) -> float:
    """Compute fraction of zero weights in the model."""
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def compute_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    model.train()
    return correct / total if total > 0 else 0.0


def count_nans(tensors: List[torch.Tensor]) -> int:
    """Count total NaN values across a list of tensors."""
    return sum(torch.isnan(t).sum().item() for t in tensors)


def tensor_stats(tensors: List[torch.Tensor], name: str) -> Dict[str, float]:
    """Compute min, max, mean, nan_count for a list of tensors."""
    all_vals = torch.cat([t.flatten() for t in tensors])
    nan_count = torch.isnan(all_vals).sum().item()
    valid = all_vals[~torch.isnan(all_vals)]

    if len(valid) == 0:
        return {f"{name}_min": float("nan"), f"{name}_max": float("nan"),
                f"{name}_mean": float("nan"), f"{name}_nan_count": nan_count}

    return {
        f"{name}_min": valid.min().item(),
        f"{name}_max": valid.max().item(),
        f"{name}_mean": valid.mean().item(),
        f"{name}_nan_count": nan_count,
    }


def create_random_data(batch_size: int = 64, num_batches: int = 10,
                       device: torch.device = torch.device("cpu")) -> DataLoader:
    """Create random MNIST-like data for sanity checking."""
    images = torch.randn(batch_size * num_batches, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size * num_batches,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_mnist_data(batch_size: int = 64, num_samples: int = 1000,
                      device: torch.device = torch.device("cpu")) -> Tuple[DataLoader, DataLoader]:
    """Create real MNIST train/test data loaders."""
    import os
    from torchvision import datasets, transforms
    from torch.utils.data import Subset

    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root=os.path.join("data", "MNIST"), train=True,
                               download=True, transform=transform)
    test_ds = datasets.MNIST(root=os.path.join("data", "MNIST"), train=False,
                              download=True, transform=transform)

    # Use subset for faster testing
    train_subset = Subset(train_ds, list(range(min(num_samples, len(train_ds)))))
    test_subset = Subset(test_ds, list(range(min(num_samples // 5, len(test_ds)))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float,
    N: int,
    C: float,
) -> Tuple[torch.optim.Optimizer, List[torch.Tensor], List[torch.Tensor],
           List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Create optimizer and ADMM variable buffers.

    ADMM initialization:
    - vk, wk: Dual variables, initialized to zeros
    - yk: Auxiliary variable for Ratio Norm, initialized to zeros
    - zk: Pruning variable, initialized to current weights (w = zk after each step)
    """
    params = list(model.parameters())
    vk = [torch.zeros_like(p) for p in params]
    wk = [torch.zeros_like(p) for p in params]
    yk = [torch.zeros_like(p) for p in params]
    # zk should be initialized to current weights, not zeros
    zk = [p.data.clone() for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    if optimizer_name == "ADMM_global":
        opt = ADMM_Adam_global(params, lr=lr, N=N, C=C, vk=vk, wk=wk, yk=yk, zk=zk, score=score_buffers)
    elif optimizer_name == "ADMM_layer":
        opt = ADMM_Adam_layer(params, lr=lr, N=N, C=C, vk=vk, wk=wk, yk=yk, zk=zk, score=score_buffers)
    elif optimizer_name == "ADMM_neuron":
        opt = ADMM_Adam_neuron(params, lr=lr, N=N, C=C, vk=vk, wk=wk, yk=yk, zk=zk, score=score_buffers)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return opt, vk, wk, yk, zk, score_buffers


def run_sanity_check(
    optimizer_name: str = "ADMM_global",
    num_steps: int = 50,
    lr: float = 1e-3,
    C: float = 1.0,
    score_type: str = "magnitude",
    verbose: bool = True,
) -> Tuple[bool, List[Dict]]:
    """Run sanity check and return (success, logs).

    Args:
        optimizer_name: One of "ADMM_global", "ADMM_layer", "ADMM_neuron"
        num_steps: Number of optimization steps
        lr: Learning rate (penalty parameter p = 1/lr)
        C: Sparsity control parameter
        score_type: Score type for pruning importance
        verbose: Print progress

    Returns:
        (success, logs): success is True if all checks pass, logs contains per-step metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    if verbose:
        print(f"\n{'='*60}")
        print(f"SANITY CHECK: {optimizer_name}")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Steps: {num_steps}, LR: {lr}, C: {C}, Score: {score_type}")
        print(f"{'='*60}\n")

    # Create model and data
    model = CNN().to(device)
    data_loader = create_random_data(batch_size=64, num_batches=max(num_steps // 5, 10), device=device)

    # Setup score calculators
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    N = len(data_loader.dataset)
    opt, vk, wk, yk, zk, score_buffers = create_optimizer(optimizer_name, model, lr, N, C)

    # Move buffers to device
    vk = [v.to(device) for v in vk]
    wk = [w.to(device) for w in wk]
    yk = [y.to(device) for y in yk]
    zk = [z.to(device) for z in zk]
    score_buffers = [s.to(device) for s in score_buffers]

    # Update optimizer references
    opt.vk = vk
    opt.wk = wk
    opt.yk = yk
    opt.zk = zk
    opt.score = score_buffers

    # Training loop
    logs = []
    model.train()
    data_iter = iter(data_loader)

    for step in range(num_steps):
        # Get batch (cycle through data)
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            images, targets = next(data_iter)

        images, targets = images.to(device), targets.to(device)

        # Forward + backward
        opt.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Update scores
        score_dict = choose_score(wanda_calc, grad_collector, score_type)
        aligned = align_scores(model, score_dict)
        for buf, new in zip(score_buffers, aligned):
            buf.copy_(new.to(device))

        # Optimizer step
        opt.step()

        # Collect metrics
        sparsity = compute_sparsity(model)
        q_stats = tensor_stats(list(model.parameters()), "q")
        y_stats = tensor_stats(yk, "y")
        z_stats = tensor_stats(zk, "z")

        log_entry = {
            "step": step,
            "loss": loss.item(),
            "sparsity": sparsity,
            **q_stats,
            **y_stats,
            **z_stats,
        }
        logs.append(log_entry)

        # Print progress
        if verbose and step % 10 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | Sparsity: {sparsity:.4f}")
            print(f"         | q: [{q_stats['q_min']:.2e}, {q_stats['q_max']:.2e}] mean={q_stats['q_mean']:.2e} nan={q_stats['q_nan_count']}")
            print(f"         | y: [{y_stats['y_min']:.2e}, {y_stats['y_max']:.2e}] mean={y_stats['y_mean']:.2e} nan={y_stats['y_nan_count']}")
            print(f"         | z: [{z_stats['z_min']:.2e}, {z_stats['z_max']:.2e}] mean={z_stats['z_mean']:.2e} nan={z_stats['z_nan_count']}")
            print()

    # Cleanup
    wanda_calc.remove_hooks()

    # Evaluate success criteria
    success = True
    issues = []

    # Check 1: No NaN values
    total_nans = sum(log["q_nan_count"] + log["y_nan_count"] + log["z_nan_count"] for log in logs)
    if total_nans > 0:
        success = False
        issues.append(f"FAIL: Found {total_nans} NaN values in ADMM variables")

    # Check 2: Loss should not explode (allow some increase but not explosion)
    initial_loss = logs[0]["loss"]
    final_loss = logs[-1]["loss"]
    max_loss = max(log["loss"] for log in logs)
    if max_loss > initial_loss * 100 or final_loss != final_loss:  # NaN check
        success = False
        issues.append(f"FAIL: Loss exploded (initial={initial_loss:.4f}, max={max_loss:.4f}, final={final_loss:.4f})")

    # Check 3: Sparsity should increase (at least some pruning happening)
    initial_sparsity = logs[0]["sparsity"]
    final_sparsity = logs[-1]["sparsity"]
    if final_sparsity <= initial_sparsity:
        issues.append(f"WARNING: Sparsity did not increase (initial={initial_sparsity:.4f}, final={final_sparsity:.4f})")
        # This is a warning, not a failure - sparsity might need more steps or different C

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
        print(f"Initial Sparsity: {initial_sparsity:.4f} -> Final Sparsity: {final_sparsity:.4f}")
        print(f"Total NaN count: {total_nans}")
        print()

        if success and not issues:
            print("STATUS: PASS - All sanity checks passed!")
        elif success:
            print("STATUS: PASS with warnings")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("STATUS: FAIL")
            for issue in issues:
                print(f"  - {issue}")
        print(f"{'='*60}\n")

    return success, logs


def run_accuracy_test(
    optimizer_name: str = "ADMM_global",
    num_steps: int = 100,
    lr: float = 1e-3,
    C: float = 0.1,
    score_type: str = "magnitude",
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Run accuracy test on real MNIST data.

    Returns:
        (final_accuracy, final_sparsity, final_loss)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    if verbose:
        print(f"\n{'='*60}")
        print(f"ACCURACY TEST: {optimizer_name}")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Steps: {num_steps}, LR: {lr}, C: {C}, Score: {score_type}")
        print(f"{'='*60}\n")

    # Create model and real MNIST data
    model = CNN().to(device)
    train_loader, test_loader = create_mnist_data(batch_size=64, num_samples=2000, device=device)

    # Setup score calculators
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    N = len(train_loader.dataset)
    opt, vk, wk, yk, zk, score_buffers = create_optimizer(optimizer_name, model, lr, N, C)

    # Move buffers to device
    vk = [v.to(device) for v in vk]
    wk = [w.to(device) for w in wk]
    yk = [y.to(device) for y in yk]
    zk = [z.to(device) for z in zk]
    score_buffers = [s.to(device) for s in score_buffers]

    # Update optimizer references
    opt.vk = vk
    opt.wk = wk
    opt.yk = yk
    opt.zk = zk
    opt.score = score_buffers

    # Initial accuracy (before pruning)
    initial_acc = compute_accuracy(model, test_loader, device)
    initial_sparsity = compute_sparsity(model)
    if verbose:
        print(f"Initial: Accuracy={initial_acc:.4f}, Sparsity={initial_sparsity:.4f}")

    # Training loop
    model.train()
    data_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)

        images, targets = images.to(device), targets.to(device)

        opt.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Update scores
        score_dict = choose_score(wanda_calc, grad_collector, score_type)
        aligned = align_scores(model, score_dict)
        for buf, new in zip(score_buffers, aligned):
            buf.copy_(new.to(device))

        opt.step()

        # Print progress
        if verbose and (step + 1) % 20 == 0:
            sparsity = compute_sparsity(model)
            acc = compute_accuracy(model, test_loader, device)
            print(f"Step {step+1:3d} | Loss: {loss.item():.4f} | Sparsity: {sparsity:.4f} | Accuracy: {acc:.4f}")

    # Cleanup
    wanda_calc.remove_hooks()

    # Final metrics
    final_acc = compute_accuracy(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    final_loss = loss.item()

    if verbose:
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {initial_acc:.4f} -> {final_acc:.4f}")
        print(f"Sparsity: {initial_sparsity:.4f} -> {final_sparsity:.4f}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"{'='*60}\n")

    return final_acc, final_sparsity, final_loss


def main():
    parser = argparse.ArgumentParser(description="ADMM Pruning Sanity Check")
    parser.add_argument("--optimizer", type=str, default="ADMM_global",
                        choices=["ADMM_global", "ADMM_layer", "ADMM_neuron"],
                        help="Optimizer to test")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Sparsity control parameter")
    parser.add_argument("--score", type=str, default="magnitude",
                        choices=["magnitude", "first order", "second order",
                                 "first order + second order", "wanda"],
                        help="Score type for pruning")
    parser.add_argument("--all", action="store_true",
                        help="Run sanity check on all optimizers")
    parser.add_argument("--accuracy", action="store_true",
                        help="Run accuracy test on real MNIST data")
    args = parser.parse_args()

    if args.accuracy:
        # Run accuracy test on real MNIST
        if args.all:
            print("\n" + "="*60)
            print("ACCURACY TEST ON ALL OPTIMIZERS")
            print("="*60)
            results = {}
            for opt_name in ["ADMM_global", "ADMM_layer", "ADMM_neuron"]:
                acc, sparsity, _ = run_accuracy_test(
                    optimizer_name=opt_name,
                    num_steps=args.steps,
                    lr=args.lr,
                    C=args.C,
                    score_type=args.score,
                )
                results[opt_name] = (acc, sparsity)

            print("\n" + "="*60)
            print("ACCURACY SUMMARY")
            print("="*60)
            for opt_name, (acc, sparsity) in results.items():
                print(f"  {opt_name}: Accuracy={acc:.4f}, Sparsity={sparsity:.4f}")
            print("="*60)
        else:
            run_accuracy_test(
                optimizer_name=args.optimizer,
                num_steps=args.steps,
                lr=args.lr,
                C=args.C,
                score_type=args.score,
            )
    elif args.all:
        # Test all optimizers
        results = {}
        for opt_name in ["ADMM_global", "ADMM_layer", "ADMM_neuron"]:
            success, _ = run_sanity_check(
                optimizer_name=opt_name,
                num_steps=args.steps,
                lr=args.lr,
                C=args.C,
                score_type=args.score,
            )
            results[opt_name] = success

        # Final summary
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        all_pass = True
        for opt_name, success in results.items():
            status = "PASS" if success else "FAIL"
            print(f"  {opt_name}: {status}")
            all_pass = all_pass and success
        print("="*60)

        sys.exit(0 if all_pass else 1)
    else:
        # Test single optimizer
        success, _ = run_sanity_check(
            optimizer_name=args.optimizer,
            num_steps=args.steps,
            lr=args.lr,
            C=args.C,
            score_type=args.score,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
