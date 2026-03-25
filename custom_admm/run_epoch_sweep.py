#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Epoch sweep experiment: Test if increasing epochs improves low-dimensional performance.

Hypothesis: The new architecture (latent=58, layers=5) needs more epochs to converge
on low-dimensional tasks (m < 512).

Usage:
    python run_epoch_sweep.py
"""

import os
import sys
import json
import time
import numpy as np
import torch

CUSTOM_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src",
)
sys.path.insert(0, CUSTOM_SRC)

from src.admm_input_group_wrapper import run_admm_input_group
from src.nn_wrapper import Model

# ============== Configuration ==============
SEED = 0
N_SAMPLES = 1000

# Test low-dimensional tasks where new architecture failed
DIMENSIONS = [16, 32, 64, 128, 256, 512, 1024]

# Epoch configurations to test
EPOCH_CONFIGS = [
    {"epochs": 500,  "warmup": 120,  "label": "500 (default)"},
    {"epochs": 1000, "warmup": 250,  "label": "1000 (2x)"},
    {"epochs": 2000, "warmup": 500,  "label": "2000 (4x)"},
]

DATASETS = ["xor", "ring"]


def generate_xor_dataset(n_samples, n_features, seed):
    """Generate XOR dataset with canonical formula."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = np.random.rand(n_samples, n_features).astype(np.float32)
    # Canonical XOR: (x1-0.5)*(0.5-x2) >= 0
    xor_mask = (X[:, 0] - 0.5) * (0.5 - X[:, 1]) >= 0
    y = xor_mask.astype(np.float32)

    return X, y


def generate_ring_dataset(n_samples, n_features, seed):
    """Generate Ring dataset."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radius = np.random.uniform(0.5, 1.0, n_samples)
    x1 = radius * np.cos(angles)
    x2 = radius * np.sin(angles)

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    X[:, 0] = x1
    X[:, 1] = x2

    # Binary labels based on distance from center
    center = np.mean(X, axis=0)
    distances = np.linalg.norm(X - center, axis=1)
    median_dist = np.median(distances)
    y = (distances > median_dist).astype(np.float32)

    return X, y


def evaluate_bestk(scores, k):
    """Calculate best-k metric."""
    top_idx = set(np.argsort(np.abs(scores))[-k:])
    true_set = set(range(k))
    return len(top_idx & true_set) / k


def run_single_experiment(dataset_name, n_features, epochs_config, seed=SEED):
    """Run a single experiment and return results."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate data
    if dataset_name == "xor":
        X, y = generate_xor_dataset(N_SAMPLES, n_features, seed)
        k = 2
    else:  # ring
        X, y = generate_ring_dataset(N_SAMPLES, n_features, seed)
        k = 2

    # Train/test split
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Run training
    t0 = time.time()

    # Pass epoch config via hp_overrides
    hp_overrides = {
        "epochs": epochs_config["epochs"],
        "warmup_epochs": epochs_config["warmup"],
    }

    y_train_hat, y_test_hat, scores, scores2 = run_admm_input_group(
        X_train_s, y_train, X_test_s, 2,  # n_classes=2
        hp_overrides=hp_overrides,
        seed=seed,
    )
    elapsed = time.time() - t0

    # Calculate metrics
    best_k = evaluate_bestk(scores, k)

    # Note: run_admm_input_group doesn't return model, so we can't get convergence info
    convergence_info = None

    return {
        "best_k": best_k,
        "runtime_seconds": elapsed,
        "non_zero_features": int(np.sum(np.abs(scores) > 1e-6)),
        "convergence": convergence_info,
    }


def main():
    """Run epoch sweep experiment."""
    print("=" * 70)
    print("  EPOCH SWEEP EXPERIMENT")
    print("  Testing: Does increasing epochs improve low-dimensional performance?")
    print("=" * 70)
    print()

    results = {
        "metadata": {
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "architectures": {
                "latent_size": 58,
                "n_hidden_layers": 5,
                "feat_drop": 0.6,
            }
        },
        "experiments": {},
    }

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*60}")

        results["experiments"][dataset_name] = {}

        for n_features in DIMENSIONS:
            print(f"\n  m={n_features}:")
            print(f"  {'-'*50}")

            results["experiments"][dataset_name][n_features] = {}

            for config in EPOCH_CONFIGS:
                label = config["label"]
                print(f"    Epochs {label}...", end=" ")

                result = run_single_experiment(dataset_name, n_features, config)

                results["experiments"][dataset_name][n_features][label] = {
                    "best_k": result["best_k"],
                    "runtime": result["runtime_seconds"],
                    "non_zero": result["non_zero_features"],
                    "convergence": result["convergence"],
                }

                print(f"best-k={result['best_k']:.1%} ({result['runtime_seconds']:.0f}s)")

                # Save intermediate results
                with open("epoch_sweep_results.json", "w") as f:
                    json.dump(results, f, indent=2, default=str)

    # Final save
    with open("epoch_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Print summary table
    for dataset_name in DATASETS:
        print(f"\n{dataset_name.upper()}:")
        print(f"{'m':>6} | {'500 (def)':>10} | {'1000 (2x)':>10} | {'2000 (4x)':>10}")
        print(f"{'-'*6}---{'-'*12}---{'-'*12}---{'-'*12}")

        for n_features in DIMENSIONS:
            row = f"{n_features:>6} | "
            for config in EPOCH_CONFIGS:
                bk = results["experiments"][dataset_name][n_features][config["label"]]["best_k"]
                row += f"{bk:>8.1%}  | "
            print(row)

    print()
    print(f"Results saved to: epoch_sweep_results.json")
    print()

    return results


if __name__ == "__main__":
    main()
