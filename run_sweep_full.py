"""Comprehensive-but-small sweep over pruning methods/hyperparameters.

Runs quick experiments on MNIST subsets to compare:
- optimizer: ADMM_global/ADMM_layer/ADMM_neuron, Ppercent_global/Ppercent_layer/Ppercent_neuron, Lasso_global/Lasso_layer/Lasso_neuron
- score: first order, second order, or first order + second order
- C grid (for ADMM & Lasso), p grid (for Ppercent), lr grid

Results are written to CSV for easy analysis. Defaults are small/fast; adjust
train_steps/epochs/val_size for deeper runs.

Example:
    python run_sweep_full.py \
        --optimizers ADMM_global ADMM_layer ADMM_neuron Ppercent_global Ppercent_layer Ppercent_neuron Lasso_global Lasso_layer Lasso_neuron \
        --score-names "first order" "second order" "first order + second order" \
        --c-values 0.5 1.0 \
        --p-values 20 40 \
        --lr-values 0.001 0.0005 \
        --train-steps 120 --epochs 1 --val-size 1000 \
        --outfile results/sweeps/full_sweep.csv
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
from optimizer.ADMM_layer import ADMM_Adam_Layer
from optimizer.ADMM_neuron import ADMM_Adam_neuron
from optimizer.ppercent_global import Ppercent_global
from optimizer.ppercent_layer import Ppercent_layer
from optimizer.ppercent_neuron import Ppercent_neuron
from optimizer.lasso_global import Lasso_global
from optimizer.lasso_layer import Lasso_layer
from optimizer.lasso_neuron import Lasso_neuron
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


def make_dataloaders(val_size: int):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=os.path.join("data", "MNIST"), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=os.path.join("data", "MNIST"), train=False, download=True, transform=transform)

    train_idx = list(range(0, 1024))  # slightly larger than demo but still small
    val_idx = list(range(0, min(val_size, len(test_ds))))
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(test_ds, val_idx), batch_size=128, shuffle=False)
    return train_loader, val_loader, len(train_ds)


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
    return correct / total if total > 0 else 0.0


def run_admm(device, lr: float, c_val: float, train_loader, val_loader, N_total: int,
             train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    zk_init = [p.clone().detach() for p in params]  # Initialize zk with weights
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = ADMM_Adam_global(
        params,
        lr=lr,
        N=N_total,
        C=c_val,
        vk=[z.clone() for z in zeros_like],
        wk=[z.clone() for z in zeros_like],
        yk=[p.clone().detach() for p in params],
        zk=zk_init,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def run_admm_layer(device, lr: float, c_val: float, train_loader, val_loader, N_total: int,
                   train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    zk_init = [p.clone().detach() for p in params]  # Initialize zk with weights
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = ADMM_Adam_Layer(
        params,
        lr=lr,
        N=N_total,
        C=c_val,
        vk=[z.clone() for z in zeros_like],
        wk=[z.clone() for z in zeros_like],
        yk=[p.clone().detach() for p in params],
        zk=zk_init,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def run_admm_neuron(device, lr: float, c_val: float, train_loader, val_loader, N_total: int,
                    train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    zk_init = [p.clone().detach() for p in params]  # Initialize zk with weights
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = ADMM_Adam_neuron(
        params,
        lr=lr,
        N=N_total,
        C=c_val,
        vk=[z.clone() for z in zeros_like],
        wk=[z.clone() for z in zeros_like],
        yk=[p.clone().detach() for p in params],
        zk=zk_init,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def run_ppercent_neuron(device, lr: float, p: float, train_loader, val_loader, N_total: int,
                        train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    score_buffers = [torch.ones_like(p_) for p_ in params]

    optimizer = Ppercent_neuron(
        params,
        lr=lr,
        p=p,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def run_ppercent_layer(device, lr: float, p: float, train_loader, val_loader, N_total: int,
                       train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    score_buffers = [torch.ones_like(p_) for p_ in params]

    optimizer = Ppercent_layer(
        params,
        lr=lr,
        p=p,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def run_ppercent_global(device, lr: float, p: float, train_loader, val_loader, N_total: int,
                        train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    score_buffers = [torch.ones_like(p_) for p_ in params]

    optimizer = Ppercent_global(
        params,
        lr=lr,
        p=p,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def run_lasso_global(device, lr: float, c_val: float, train_loader, val_loader, N_total: int,
                     train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    vk = [torch.zeros_like(p) for p in params]
    zk = [torch.zeros_like(p) for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Lasso_global(
        params,
        lr=lr,
        N=N_total,
        C=c_val,
        vk=vk,
        zk=zk,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def run_lasso_neuron(device, lr: float, c_val: float, train_loader, val_loader, N_total: int,
                     train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    vk = [torch.zeros_like(p) for p in params]
    zk = [torch.zeros_like(p) for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Lasso_neuron(
        params,
        lr=lr,
        N=N_total,
        C=c_val,
        vk=vk,
        zk=zk,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def run_lasso_layer(device, lr: float, c_val: float, train_loader, val_loader, N_total: int,
                    train_steps: int, epochs: int, score_name: str):
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    vk = [torch.zeros_like(p) for p in params]
    zk = [torch.zeros_like(p) for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Lasso_layer(
        params,
        lr=lr,
        N=N_total,
        C=c_val,
        vk=vk,
        zk=zk,
        score=score_buffers,
    )

    model.train()
    step_count = 0
    for _ in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                buf.copy_(torch.clamp(new, min=1e-3))

            optimizer.step()
            step_count += 1
            if step_count >= train_steps:
                break
        if step_count >= train_steps:
            break

    val_acc = evaluate(model, val_loader, device)
    sparsity = compute_sparsity(model)
    wanda_calc.remove_hooks()
    return val_acc, sparsity, step_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=[
            "ADMM_global",
            "ADMM_layer",
            "ADMM_neuron",
            "Ppercent_global",
            "Ppercent_layer",
            "Ppercent_neuron",
            "Lasso_global",
            "Lasso_layer",
            "Lasso_neuron",
        ],
        choices=[
            "ADMM_global",
            "ADMM_layer",
            "ADMM_neuron",
            "Ppercent_global",
            "Ppercent_layer",
            "Ppercent_neuron",
            "Lasso_global",
            "Lasso_layer",
            "Lasso_neuron",
        ],
    )
    parser.add_argument("--score-names", nargs="+", default=["first order", "second order", "first order + second order"],
                        help="Score names passed to choose_score")
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0])  # for ADMM & Lasso
    parser.add_argument("--p-values", nargs="+", type=float, default=[10, 20, 40, 50])    # for Ppercent
    parser.add_argument("--lr-values", nargs="+", type=float, default=[1e-3, 5e-4])
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--outfile", type=str, default="results/sweeps/full_sweep.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    train_loader, val_loader, N_total = make_dataloaders(args.val_size)
    N_total = len(train_loader.dataset)

    rows = []
    for optimizer_name in args.optimizers:
        for score_name in args.score_names:
            for lr in args.lr_values:
                if optimizer_name == "ADMM_global":
                    for c_val in args.c_values:
                        val_acc, sparsity, steps = run_admm(
                            device, lr, c_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": c_val,
                            "p": None,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} C={c_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")
                elif optimizer_name == "ADMM_layer":
                    for c_val in args.c_values:
                        val_acc, sparsity, steps = run_admm_layer(
                            device, lr, c_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": c_val,
                            "p": None,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} C={c_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")
                elif optimizer_name == "ADMM_neuron":
                    for c_val in args.c_values:
                        val_acc, sparsity, steps = run_admm_neuron(
                            device, lr, c_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": c_val,
                            "p": None,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} C={c_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")
                elif optimizer_name == "Ppercent_global":
                    for p_val in args.p_values:
                        val_acc, sparsity, steps = run_ppercent_global(
                            device, lr, p_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": None,
                            "p": p_val,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} p={p_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")
                elif optimizer_name == "Ppercent_layer":
                    for p_val in args.p_values:
                        val_acc, sparsity, steps = run_ppercent_layer(
                            device, lr, p_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": None,
                            "p": p_val,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} p={p_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")
                elif optimizer_name == "Ppercent_neuron":
                    for p_val in args.p_values:
                        val_acc, sparsity, steps = run_ppercent_neuron(
                            device, lr, p_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": None,
                            "p": p_val,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} p={p_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")
                elif optimizer_name == "Lasso_global":
                    for c_val in args.c_values:
                        val_acc, sparsity, steps = run_lasso_global(
                            device, lr, c_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": c_val,
                            "p": None,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} C={c_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")
                elif optimizer_name == "Lasso_layer":
                    for c_val in args.c_values:
                        val_acc, sparsity, steps = run_lasso_layer(
                            device, lr, c_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": c_val,
                            "p": None,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} C={c_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")
                elif optimizer_name == "Lasso_neuron":
                    for c_val in args.c_values:
                        val_acc, sparsity, steps = run_lasso_neuron(
                            device, lr, c_val, train_loader, val_loader, N_total, args.train_steps, args.epochs, score_name
                        )
                        rows.append({
                            "optimizer": optimizer_name,
                            "score": score_name,
                            "lr": lr,
                            "C": c_val,
                            "p": None,
                            "val_acc": val_acc,
                            "sparsity": sparsity,
                            "train_steps": steps,
                        })
                        print(f"opt={optimizer_name} score={score_name} C={c_val} lr={lr} val_acc={val_acc:.4f} sparsity={sparsity:.3f}")

    with open(args.outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "score", "lr", "C", "p", "val_acc", "sparsity", "train_steps"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved sweep results to {args.outfile}")


if __name__ == "__main__":
    main()
