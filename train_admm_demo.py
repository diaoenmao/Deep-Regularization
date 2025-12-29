"""Minimal ADMM pruning demo on MNIST (small subset).

This is a reference script to show how the refactored components work together.
It runs a few steps only; tweak hyperparameters and dataset size for real training.
"""
from __future__ import annotations

import math
import os
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


def align_scores(model: nn.Module, score_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Return scores ordered to match model.parameters()."""
    ordered = []
    for name, _ in model.named_parameters():
        if name not in score_dict:
            raise KeyError(f"Missing score for parameter: {name}")
        ordered.append(score_dict[name])
    return ordered


def compute_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=os.path.join("data", "MNIST"), train=True, download=True, transform=transform)
    # small subset for demo speed
    idx = list(range(0, 512))
    train_subset = Subset(train_ds, idx)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    # Initialize buffers for ADMM variables and scores
    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    score_buffers = [torch.ones_like(p) for p in params]  # updated each step

    optimizer = ADMM_Adam_global(
        params,
        lr=1e-3,
        N=len(train_ds),
        C=1.0,
        vk=zeros_like,
        wk=zeros_like,
        yk=zeros_like,
        zk=zeros_like,
        score=score_buffers,
    )

    model.train()
    for step, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # compute scores from current activations/gradients
        score_dict = choose_score(wanda_calc, grad_collector, "second order")
        aligned = align_scores(model, score_dict)
        for buf, new in zip(score_buffers, aligned):
            buf.copy_(new)

        optimizer.step()

        if step % 10 == 0:
            sparsity = compute_sparsity(model)
            print(f"step={step:03d} loss={loss.item():.4f} sparsity={sparsity:.3f}")

        if step >= 30:
            break

    wanda_calc.remove_hooks()


if __name__ == "__main__":
    main()
