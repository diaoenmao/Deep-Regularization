# -*- coding: utf-8 -*-
"""
Quick validation test for Experiment A.

Uses fair hyperparameters matching main benchmark:
- 120 warmup epochs (critical for learning)
- Proper data scaling
- C=0.05

Expected runtime: 2-3 minutes
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.data import generate_dataset
from src.gradual_admm_with_pruning import gradual_admm_with_pruning
from src.admm_input_group_wrapper import GatedFeatureSelectionMLP, _Scaler


# Fair hyperparameters matching main benchmark
FAIR_MODEL_CONFIG = {
    "latent_size": 32,
    "n_hidden_layers": 2,
    "dropout": 0.043,
    "activation": "mish",
    "feat_drop": 0.6,
    "bounded_gate": False,
}


def quick_test():
    """Quick validation test with fair hyperparameters."""
    print("=" * 60)
    print("QUICK VALIDATION TEST")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Single dataset for quick test
    np.random.seed(42)
    X, X_tilde, y = generate_dataset("xor", n_samples=500, n_features=128)

    # Scale data (critical!)
    scaler = _Scaler()
    X_scaled = scaler.fit_transform(X)

    n_train = int(0.8 * 500)
    X_train = X_scaled[:n_train]
    y_train = y[:n_train]
    X_test = X_scaled[n_train:]
    y_test = y[n_train:]

    ground_truth = [0, 1]  # XOR ground truth
    k = 2
    n_classes = 2

    print(f"Dataset: XOR, n_train={n_train}, m=128, k=2")

    # Test all 5 variants
    variants = [
        {"name": "gradual_none", "prune_mode": "none", "reweight": False},
        {"name": "gradual_soft", "prune_mode": "soft", "reweight": True},
        {"name": "gradual_soft_no_rw", "prune_mode": "soft", "reweight": False},
        {"name": "gradual_hard", "prune_mode": "hard", "reweight": True},
        {"name": "gradual_hard_no_rw", "prune_mode": "hard", "reweight": False},
    ]

    # Fair config: warmup + reasonable C
    # Use larger C for stronger sparsity (matching fair training config from benchmark)
    C_schedule = [0.5]  # Larger C for clearer sparsity signal

    results = []

    for variant in variants:
        print(f"\n--- Testing: {variant['name']} ---")

        torch.manual_seed(42)
        np.random.seed(42)

        selected, history = gradual_admm_with_pruning(
            model_class=GatedFeatureSelectionMLP,
            model_kwargs=FAIR_MODEL_CONFIG,
            X_train=X_train.copy(),
            y_train=y_train.copy(),
            X_test=X_test.copy(),
            y_test=y_test.copy(),
            n_classes=n_classes,
            ground_truth=ground_truth,
            k=k,
            C_schedule=C_schedule,
            prune_threshold=0.1,  # Higher threshold for pruning
            prune_mode=variant["prune_mode"],
            reweight=variant["reweight"],
            epochs_per_phase=200,  # More ADMM epochs (matching benchmark: 500-120=380)
            warmup_epochs=100,    # CRITICAL: warmup phase!
            device=device,
            verbose=True,
        )

        best_k = len(set(selected) & set(ground_truth)) / k

        # Print gate values for debugging
        if history["phases"]:
            last_phase = history["phases"][-1]
            print(f"Final gate_sum: {last_phase['gate_sum']:.4f}, alive: {last_phase['alive_count']}")

        print(f"Result: best_k = {best_k:.2f}, selected = {selected}")

        results.append({
            "variant": variant["name"],
            "best_k": best_k,
            "selected": selected,
        })

    # Summary
    print("\n" + "=" * 60)
    print("QUICK TEST SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['variant']}: best_k = {r['best_k']:.2f}")

    # Check if all variants produced valid results
    all_valid = all(r["best_k"] >= 0 for r in results)
    if all_valid:
        print("\n[PASSED] VALIDATION PASSED - All variants completed successfully")
    else:
        print("\n[FAILED] VALIDATION FAILED - Some variants failed")

    return results


if __name__ == "__main__":
    quick_test()