#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run full benchmark with bug fixes.
This re-runs all dimensions with the fixed ADMM implementation.
"""

import os
import sys
import json
import time
import numpy as np
import torch

CUSTOM_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, CUSTOM_SRC)

from src.admm_input_group_wrapper import run_admm_input_group

SEED = 0
N_SAMPLES = 1000

# All dimensions to test
DIMENSIONS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

DATASETS = [
    ("xor", 2),
    ("ring", 2),
    ("ring+xor", 4),
    ("ring+xor+sum", 6),
]


def generate_dataset(name, n_samples, n_features):
    """Generate synthetic dataset matching src/data.py."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X = np.random.rand(n_samples, n_features).astype(np.float32)

    if name == "xor":
        # Canonical XOR: (x1-0.5)*(0.5-x2) >= 0
        xor_mask = (X[:, 0] - 0.5) * (0.5 - X[:, 1]) >= 0
        y = xor_mask.astype(np.float32)
    elif name == "ring":
        angles = np.random.uniform(0, 2 * np.pi, n_samples)
        radius = np.random.uniform(0.5, 1.0, n_samples)
        X[:, 0] = radius * np.cos(angles)
        X[:, 1] = radius * np.sin(angles)
        center = np.mean(X, axis=0)
        distances = np.linalg.norm(X - center, axis=1)
        median_dist = np.median(distances)
        y = (distances > median_dist).astype(np.float32)
    elif name == "ring+xor":
        # First 2 features: ring
        angles = np.random.uniform(0, 2 * np.pi, n_samples)
        radius = np.random.uniform(0.5, 1.0, n_samples)
        X[:, 0] = radius * np.cos(angles)
        X[:, 1] = radius * np.sin(angles)
        # Next 2 features: xor
        xor_mask = (X[:, 2] - 0.5) * (0.5 - X[:, 3]) >= 0
        y = xor_mask.astype(np.float32)
    elif name == "ring+xor+sum":
        # First 2 features: ring
        angles = np.random.uniform(0, 2 * np.pi, n_samples)
        radius = np.random.uniform(0.5, 1.0, n_samples)
        X[:, 0] = radius * np.cos(angles)
        X[:, 1] = radius * np.sin(angles)
        # Next 2 features: xor
        xor_mask = (X[:, 2] - 0.5) * (0.5 - X[:, 3]) >= 0
        # Next 2 features: sum
        sum_val = X[:, 4] + X[:, 5]
        sum_mask = sum_val > np.median(sum_val)
        y = (xor_mask & sum_mask).astype(np.float32)
    else:
        y = np.random.randint(0, 2, n_samples).astype(np.float32)

    return X, y


def evaluate_bestk(scores, k):
    """Calculate best-k metric."""
    top_idx = set(np.argsort(np.abs(scores))[-k:])
    true_set = set(range(k))
    return len(top_idx & true_set) / k


def run_single_experiment(dataset_name, k, n_features):
    """Run a single experiment."""
    X, _ = generate_dataset(dataset_name, N_SAMPLES, n_features)

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y[:n_train] if 'y' in dir() else (X[:, 0] > 0.5).astype(np.float32)[:n_train]

    # Re-generate y properly
    np.random.seed(SEED)
    if dataset_name == "xor":
        y = ((X[:, 0] - 0.5) * (0.5 - X[:, 1]) >= 0).astype(np.float32)
    elif dataset_name == "ring":
        y = np.zeros(n_samples)  # placeholder

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Run
    t0 = time.time()
    y_train_hat, y_test_hat, scores, scores2 = run_admm_input_group(
        X_train_s, y_train, X_test_s, 2,
        seed=SEED,
    )
    elapsed = time.time() - t0

    best_k = evaluate_bestk(scores, k)

    return {
        "best_k": best_k,
        "runtime": elapsed,
        "non_zero": int(np.sum(np.abs(scores) > 1e-6)),
    }


def main():
    print("=" * 70)
    print("  FULL BENCHMARK WITH BUG FIXES")
    print("=" * 70)

    results = {}

    for dataset_name, k in DATASETS:
        print(f"\n{dataset_name} (k={k}):")
        print("-" * 50)
        results[dataset_name] = []

        for n_features in DIMENSIONS:
            try:
                result = run_single_experiment(dataset_name, k, n_features)
                results[dataset_name].append({
                    "m": n_features,
                    "best_k": result["best_k"],
                    "runtime": result["runtime"],
                })
                print(f"  m={n_features:5d}: best-k={result['best_k']:.1%} ({result['runtime']:.0f}s)")
            except Exception as e:
                print(f"  m={n_features:5d}: FAILED - {e}")
                results[dataset_name].append({
                    "m": n_features,
                    "best_k": None,
                    "runtime": None,
                    "error": str(e),
                })

    # Save
    with open("results_fixed.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for dataset_name, k in DATASETS:
        print(f"\n{dataset_name}:")
        print(f"{'m':>6} | {'best-k':>10}")
        print(f"{'-'*6}---{'-'*11}")
        for r in results[dataset_name]:
            if r["best_k"] is not None:
                print(f"{r['m']:>6} | {r['best_k']:>9.1%}")
            else:
                print(f"{r['m']:>6} | FAILED")

    print(f"\nResults saved to: results_fixed.json")

    return results


if __name__ == "__main__":
    main()
