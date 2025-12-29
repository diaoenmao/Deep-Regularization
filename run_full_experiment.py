"""Full-scale experiment for pruning methods comparison.

This script runs comprehensive experiments (~10 hours) across:
- 9 optimizers: ADMM/Ppercent/Lasso × global/layer/neuron
- 3 score types: first order, second order, first order + second order
- Multiple sparsity levels (controlled by C or p parameter)
- Full MNIST dataset with proper train/val/test splits

Output: JSON file compatible with plotting scripts, CSV for analysis.

Usage:
    python run_full_experiment.py --epochs 10 --device cuda
    python run_full_experiment.py --epochs 5 --device cpu  # shorter run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

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
from Score.score_choos import choose_score, normalize_scores


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(batch_size: int = 64):
    """Get full MNIST train/test loaders."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data/MNIST", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data/MNIST", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)
    return train_loader, test_loader, len(train_ds)


def align_scores(model: nn.Module, score_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Align score dict to parameter order."""
    ordered = []
    for name, _ in model.named_parameters():
        ordered.append(score_dict[name])
    return ordered


def compute_sparsity(model: nn.Module) -> float:
    """Compute fraction of zero weights."""
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def compute_pq_index(model: nn.Module) -> float:
    """Compute PQ index (weight distribution metric)."""
    all_weights = []
    for p in model.parameters():
        all_weights.append(p.data.abs().flatten())
    all_weights = torch.cat(all_weights)
    
    if len(all_weights) == 0:
        return 0.0
    
    # PQ index: ratio of L1 norm to L2 norm (normalized)
    l1 = all_weights.sum()
    l2 = torch.sqrt((all_weights ** 2).sum())
    n = len(all_weights)
    
    if l2 == 0:
        return 0.0
    
    # Normalized PQ: 1 means uniform, closer to 0 means sparse
    pq = (l1 / l2) / (n ** 0.5)
    return pq.item()


def evaluate(model: nn.Module, loader: DataLoader, device) -> float:
    """Evaluate model accuracy."""
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


# ============================================================================
# Early Stopping Configuration
# ============================================================================

EARLY_STOP_PATIENCE = 3          # Stop if no improvement for N epochs
EARLY_STOP_MIN_ACC = 15.0        # Stop immediately if acc drops below this (collapsed model)
EARLY_STOP_MIN_DELTA = 0.1       # Minimum improvement to reset patience


def train_with_early_stopping(model, optimizer, train_loader, test_loader, device, 
                               epochs, score_name, wanda_calc, grad_collector, score_buffers):
    """Generic training loop with early stopping.
    
    Early stopping triggers:
    1. Accuracy collapses below MIN_ACC (model broken)
    2. No improvement for PATIENCE epochs (converged)
    
    Returns: (best_acc, final_sparsity, final_pq, epochs_run)
    """
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    patience_counter = 0
    epochs_run = 0
    
    # Check if we need Fisher accumulation for second-order scores
    needs_fisher = "second order" in score_name.lower()
    
    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        
        # Reset Fisher accumulator at the start of each epoch for fresh statistics
        if needs_fisher:
            grad_collector.reset_fisher()
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            # Accumulate Fisher information after backward (for second-order)
            if needs_fisher:
                grad_collector.accumulate_fisher()

            score_dict = choose_score(wanda_calc, grad_collector, score_name)
            # Normalize scores to make different score types comparable
            score_dict = normalize_scores(score_dict)
            aligned = align_scores(model, score_dict)
            for buf, new in zip(score_buffers, aligned):
                # Use smaller min clamp for second-order (values can be very small)
                buf.copy_(torch.clamp(new, min=1e-8))

            optimizer.step()
        
        # Evaluate after each epoch
        acc = evaluate(model, test_loader, device)
        
        # Early stop: model collapsed
        if acc < EARLY_STOP_MIN_ACC:
            break
        
        # Track best and patience
        if acc > best_acc + EARLY_STOP_MIN_DELTA:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stop: converged (no improvement)
        if patience_counter >= EARLY_STOP_PATIENCE:
            break
    
    # Final metrics
    final_acc = evaluate(model, test_loader, device)
    sparsity = compute_sparsity(model)
    pq = compute_pq_index(model)
    
    return max(best_acc, final_acc), sparsity, pq, epochs_run


# ============================================================================
# Training functions for each optimizer type
# ============================================================================

def train_admm_global(device, lr, c_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with ADMM global optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    zk_init = [p.clone().detach() for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = ADMM_Adam_global(
        params, lr=lr, N=N_total, C=c_val,
        vk=[z.clone() for z in zeros_like],
        wk=[z.clone() for z in zeros_like],
        yk=[p.clone().detach() for p in params],
        zk=zk_init,
        score=score_buffers,
    )

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


def train_admm_layer(device, lr, c_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with ADMM layer optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    zk_init = [p.clone().detach() for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = ADMM_Adam_Layer(
        params, lr=lr, N=N_total, C=c_val,
        vk=[z.clone() for z in zeros_like],
        wk=[z.clone() for z in zeros_like],
        yk=[p.clone().detach() for p in params],
        zk=zk_init,
        score=score_buffers,
    )

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


def train_admm_neuron(device, lr, c_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with ADMM neuron optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    zk_init = [p.clone().detach() for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = ADMM_Adam_neuron(
        params, lr=lr, N=N_total, C=c_val,
        vk=[z.clone() for z in zeros_like],
        wk=[z.clone() for z in zeros_like],
        yk=[p.clone().detach() for p in params],
        zk=zk_init,
        score=score_buffers,
    )

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


def train_ppercent_global(device, lr, p_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with Ppercent global optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Ppercent_global(params, lr=lr, p=p_val, score=score_buffers)

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


def train_ppercent_layer(device, lr, p_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with Ppercent layer optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Ppercent_layer(params, lr=lr, p=p_val, score=score_buffers)

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


def train_ppercent_neuron(device, lr, p_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with Ppercent neuron optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Ppercent_neuron(params, lr=lr, p=p_val, score=score_buffers)

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


def train_lasso_global(device, lr, c_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with Lasso global optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Lasso_global(
        params, lr=lr, N=N_total, C=c_val,
        vk=zeros_like, zk=[p.clone() for p in params],
        score=score_buffers,
    )

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


def train_lasso_layer(device, lr, c_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with Lasso layer optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Lasso_layer(
        params, lr=lr, N=N_total, C=c_val,
        vk=zeros_like, zk=[p.clone() for p in params],
        score=score_buffers,
    )

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


def train_lasso_neuron(device, lr, c_val, train_loader, test_loader, N_total, epochs, score_name):
    """Train with Lasso neuron optimizer."""
    model = CNN().to(device)
    wanda_calc = WANDA_ScoreCalculator(model)
    grad_collector = GradientCollector(model)

    params = list(model.parameters())
    zeros_like = [torch.zeros_like(p) for p in params]
    score_buffers = [torch.ones_like(p) for p in params]

    optimizer = Lasso_neuron(
        params, lr=lr, N=N_total, C=c_val,
        vk=zeros_like, zk=[p.clone() for p in params],
        score=score_buffers,
    )

    acc, sparsity, pq, epochs_run = train_with_early_stopping(
        model, optimizer, train_loader, test_loader, device,
        epochs, score_name, wanda_calc, grad_collector, score_buffers
    )
    wanda_calc.remove_hooks()
    return acc, sparsity, pq


# ============================================================================
# Experiment Configuration
# ============================================================================

# C values for ADMM and Lasso (logarithmically spaced for good coverage)
C_VALUES_ADMM_GLOBAL = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
C_VALUES_ADMM_LAYER = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
C_VALUES_ADMM_NEURON = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
C_VALUES_LASSO_GLOBAL = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
C_VALUES_LASSO_LAYER = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
C_VALUES_LASSO_NEURON = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# P values for Ppercent (target sparsity percentages)
P_VALUES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]

# Score types based on Taylor expansion framework from the paper
# - magnitude: |W| (Data-free baseline)
# - first order: |∂L/∂W · W| (Gradient × Weight)
# - second order: (1/2)Σ(∂L/∂W · W)² (Fisher approximation)
# - first order + second order: Complete Taylor expansion
SCORE_NAMES = ["magnitude", "first order", "second order", "first order + second order"]

# Map for class names (for JSON output compatibility)
CLASS_NAME_MAP = {
    "ADMM_global": "ADMM_Adam_Global",
    "ADMM_layer": "ADMM_Adam_Layer",
    "ADMM_neuron": "ADMM_Adam_Neuron",
    "Ppercent_global": "Ppercent_Adam_Global",
    "Ppercent_layer": "Ppercent_Adam_Layer",
    "Ppercent_neuron": "Ppercent_Adam_Neuron",
    "Lasso_global": "Lasso_Adam_Global",
    "Lasso_layer": "Lasso_Adam_Layer",
    "Lasso_neuron": "Lasso_Adam_Neuron",
}

SCORE_NAME_MAP = {
    "magnitude": "Magnitude",
    "first order": "First-Order",
    "second order": "Second-Order",
    "first order + second order": "First+Second-Order",
}


def run_experiment(args):
    """Run full experiment."""
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    train_loader, test_loader, N_total = get_dataloaders(args.batch_size)
    print(f"Dataset: MNIST, Train: {N_total}, Test: 10000")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print()
    
    results = []
    total_experiments = (
        3 * len(C_VALUES_ADMM_GLOBAL) * len(SCORE_NAMES) +  # ADMM global
        3 * len(C_VALUES_ADMM_LAYER) * len(SCORE_NAMES) +   # ADMM layer  
        3 * len(C_VALUES_ADMM_NEURON) * len(SCORE_NAMES) +  # ADMM neuron
        3 * len(P_VALUES) * len(SCORE_NAMES) +              # Ppercent (3 variants)
        3 * len(C_VALUES_LASSO_GLOBAL) * len(SCORE_NAMES) + # Lasso global
        3 * len(C_VALUES_LASSO_LAYER) * len(SCORE_NAMES) +  # Lasso layer
        3 * len(C_VALUES_LASSO_NEURON) * len(SCORE_NAMES)   # Lasso neuron
    )
    
    # Actually it's 9 optimizers × 3 scores × 10 param values = 270 experiments
    total_experiments = 9 * 3 * 10
    exp_count = 0
    start_time = time.time()
    
    # ========== ADMM Global ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["ADMM_global"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for c_val in C_VALUES_ADMM_GLOBAL:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] ADMM_global | {score_name} | C={c_val}")
            
            acc, sparsity, pq = train_admm_global(
                device, args.lr, c_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(c_val)
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        # Sort by remaining weights
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # ========== ADMM Layer ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["ADMM_layer"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for c_val in C_VALUES_ADMM_LAYER:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] ADMM_layer | {score_name} | C={c_val}")
            
            acc, sparsity, pq = train_admm_layer(
                device, args.lr, c_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(c_val)
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # ========== ADMM Neuron ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["ADMM_neuron"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for c_val in C_VALUES_ADMM_NEURON:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] ADMM_neuron | {score_name} | C={c_val}")
            
            acc, sparsity, pq = train_admm_neuron(
                device, args.lr, c_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(c_val)
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # ========== Ppercent Global ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["Ppercent_global"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for p_val in P_VALUES:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] Ppercent_global | {score_name} | p={p_val}")
            
            acc, sparsity, pq = train_ppercent_global(
                device, args.lr, p_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(p_val)  # Store p as C for compatibility
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # ========== Ppercent Layer ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["Ppercent_layer"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for p_val in P_VALUES:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] Ppercent_layer | {score_name} | p={p_val}")
            
            acc, sparsity, pq = train_ppercent_layer(
                device, args.lr, p_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(p_val)
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # ========== Ppercent Neuron ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["Ppercent_neuron"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for p_val in P_VALUES:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] Ppercent_neuron | {score_name} | p={p_val}")
            
            acc, sparsity, pq = train_ppercent_neuron(
                device, args.lr, p_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(p_val)
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # ========== Lasso Global ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["Lasso_global"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for c_val in C_VALUES_LASSO_GLOBAL:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] Lasso_global | {score_name} | C={c_val}")
            
            acc, sparsity, pq = train_lasso_global(
                device, args.lr, c_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(c_val)
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # ========== Lasso Layer ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["Lasso_layer"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for c_val in C_VALUES_LASSO_LAYER:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] Lasso_layer | {score_name} | C={c_val}")
            
            acc, sparsity, pq = train_lasso_layer(
                device, args.lr, c_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(c_val)
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # ========== Lasso Neuron ==========
    for score_name in SCORE_NAMES:
        result_entry = {
            "class_name": CLASS_NAME_MAP["Lasso_neuron"],
            "score_name": SCORE_NAME_MAP[score_name],
            "C": [],
            "accuracy": [],
            "remaining_weights": [],
            "pq_index": [],
        }
        
        for c_val in C_VALUES_LASSO_NEURON:
            exp_count += 1
            print(f"[{exp_count}/{total_experiments}] Lasso_neuron | {score_name} | C={c_val}")
            
            acc, sparsity, pq = train_lasso_neuron(
                device, args.lr, c_val, train_loader, test_loader, N_total, args.epochs, score_name
            )
            
            result_entry["C"].append(c_val)
            result_entry["accuracy"].append(acc)
            result_entry["remaining_weights"].append(1.0 - sparsity)
            result_entry["pq_index"].append(pq)
            
            elapsed = time.time() - start_time
            eta = elapsed / exp_count * (total_experiments - exp_count)
            print(f"    Acc: {acc:.2f}%, Sparsity: {sparsity*100:.1f}%, ETA: {eta/3600:.1f}h")
        
        sorted_indices = sorted(range(len(result_entry["remaining_weights"])), 
                               key=lambda i: result_entry["remaining_weights"][i], reverse=True)
        result_entry["save_wei_sorted"] = [result_entry["remaining_weights"][i] for i in sorted_indices]
        result_entry["save_accwei_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        result_entry["save_pq_sorted"] = [result_entry["pq_index"][i] for i in sorted_indices]
        result_entry["save_accpq_sorted"] = [result_entry["accuracy"][i] for i in sorted_indices]
        
        results.append(result_entry)
    
    # Save results
    os.makedirs("results/metrics", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/metrics/full_experiment_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Also save as the standard filename for plotting
    with open("results/metrics/cnn3_MNIST_all_optimizers_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Experiment completed in {total_time/3600:.2f} hours")
    print(f"Results saved to: {output_file}")
    print(f"Also saved to: results/metrics/cnn3_MNIST_all_optimizers_experiment_results.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Full pruning experiment")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per config")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
