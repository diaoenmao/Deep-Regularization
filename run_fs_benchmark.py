#!/usr/bin/env python
"""
Quick-run script for the Feature-Selection-Benchmark with ADMM/Lasso methods.

Usage:
    python run_fs_benchmark.py admm_global
    python run_fs_benchmark.py lasso_neuron
    python run_fs_benchmark.py --all           # run all 6 methods
    python run_fs_benchmark.py --quick admm_global  # small quick test

The script sets up sys.path and delegates to the benchmark's main-benchmark.py.
"""

import argparse
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(ROOT, "Feature-Selection-Benchmark")

ADMM_LASSO_METHODS = [
    "admm_global",
    "admm_layer",
    "admm_neuron",
    "lasso_global",
    "lasso_layer",
    "lasso_neuron",
]

ALL_METHODS = [
    "rf", "mi", "relief",
    "admm_global", "admm_layer", "admm_neuron",
    "lasso_global", "lasso_layer", "lasso_neuron",
]


def run_benchmark(method: str) -> None:
    """Run main-benchmark.py for a single method."""
    cmd = [sys.executable, "main-benchmark.py", method]
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, cwd=BENCH, check=True)


def run_quick_test(method: str) -> None:
    """Run a minimal sanity-check: one dataset, few features, 1 fold."""
    print(f"\n{'='*60}")
    print(f"  Quick test: {method}")
    print(f"{'='*60}\n")

    sys.path.insert(0, BENCH)
    sys.path.insert(0, ROOT)

    from src.data import generate_dataset
    from src.core import run_fs_method
    import numpy as np
    from sklearn.model_selection import KFold

    n_features = 32
    n_samples = 200
    dataset_name = "xor"

    X, X_tilde, y = generate_dataset(dataset_name, n_samples, n_features)
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    splits = list(KFold(n_splits=3).split(X))
    train_idx, test_idx = splits[0]
    X_train, X_test = X[train_idx], X[test_idx]
    X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    y_train_hat, y_hat, scores, scores2 = run_fs_method(
        dataset_name, method, X_train, X_tilde_train,
        y_train, X_test, X_tilde_test, k=2,
    )

    print(f"\n--- Results for {method} on {dataset_name} ({n_features} features) ---")
    if scores is not None:
        top_k_idx = np.argsort(np.abs(scores))[-2:]
        correct = set(range(2))  # XOR uses features 0,1
        hit = sum(1 for i in top_k_idx if i in correct)
        print(f"  Top-2 features: {sorted(top_k_idx.tolist())}")
        print(f"  Correct features found: {hit}/2")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    if y_hat is not None:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_test, y_hat if len(y_hat.shape) == 1 else y_hat[:, 1])
            print(f"  Test AUROC: {auc:.4f}")
        except Exception as e:
            print(f"  AUROC computation failed: {e}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Feature-Selection-Benchmark with ADMM/Lasso methods"
    )
    parser.add_argument(
        "method", nargs="?", default=None,
        help="Method name (e.g. admm_global, lasso_neuron). "
             "Use --all to run all methods."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all ADMM/Lasso methods sequentially."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a quick sanity-check instead of the full benchmark."
    )
    args = parser.parse_args()

    if args.all:
        methods = ADMM_LASSO_METHODS
    elif args.method:
        methods = [args.method]
    else:
        parser.print_help()
        sys.exit(1)

    for m in methods:
        if args.quick:
            run_quick_test(m)
        else:
            run_benchmark(m)

    print("\nAll done!")
