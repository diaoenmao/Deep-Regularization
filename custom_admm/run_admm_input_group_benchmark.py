#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run ADMM Input Group through the official Feature-Selection-Benchmark protocol.

This script:
1. Uses the official data generation from Feature-Selection-Benchmark/src/data.py
2. Uses the official evaluation protocol (6-fold CV, best-k metric)
3. Calls our custom admm_input_group implementation directly

Architecture:
    custom_admm/
    ├── run_admm_input_group_benchmark.py  (this script)
    ├── src/
    │   ├── admm_input_group_wrapper.py    (our custom method)
    │   ├── data.py                        (copy of benchmark's data.py)
    │   └── core.py                        (copy of benchmark's core.py)
    └── results/                           (output results)

Usage:
    cd custom_admm
    python run_admm_input_group_benchmark.py

Results are saved to: custom_admm/results/admm_input_group-{dataset}-{n_samples}.txt
"""
import sys
import os
import time
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Add custom_admm/src to path for importing our modules
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Import data generation (copy of benchmark's data.py - same implementation)
from data import generate_dataset

# Import our custom ADMM method directly
from admm_input_group_wrapper import run_admm_input_group

# Configuration
SEED = 0
N_SAMPLES = 1000
METHOD = "admm_input_group"

# Dataset configuration matching the official benchmark
datasets_config = [
    ("xor",          2, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring",         2, [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor",     4, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor+sum", 6, [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
]

results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)


def evaluate_fold(train_idx, test_idx, X, X_tilde, y, k, n_features):
    """
    Evaluate a single fold of cross-validation.

    Follows the official benchmark protocol:
    1. Split data into train/test
    2. Shuffle features (to test method's ability to find true features)
    3. Run feature selection method
    4. Compute best-k metric
    """
    X_train, X_test = X[train_idx], X[test_idx]
    X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Shuffle features (as per benchmark protocol)
    idx = np.arange(n_features)
    np.random.shuffle(idx)
    X_train, X_test = X_train[:, idx], X_test[:, idx]
    X_tilde_train, X_tilde_test = X_tilde_train[:, idx], X_tilde_test[:, idx]
    correct = set(np.where(idx < k)[0].tolist())

    # Run ADMM Input Group
    y_train_hat, y_hat, scores, scores2 = run_admm_input_group(
        X_train, y_train, X_test,
        n_classes=2,
        use_ratio_norm=True,
        use_admm=True,
        seed=SEED
    )

    # Compute best-k metric
    if scores is not None:
        ranked = np.argsort(np.abs(scores))
        best_k = sum(i in correct for i in ranked[-k:]) / k
    else:
        best_k = 0

    # Compute best-2k metric
    if scores2 is not None:
        ranked2 = np.argsort(np.abs(scores2))
        best_2k = sum(i in correct for i in ranked2[-(2*k):]) / k
    else:
        best_2k = 0

    return best_k, best_2k


def run_benchmark():
    """Run full benchmark for all datasets and dimensions."""
    print("=" * 70)
    print("  ADMM Input Group - Feature Selection Benchmark")
    print("=" * 70)
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  N_folds: 6")
    print(f"  Method: ADMM Input Group + Ratio Norm")
    print("=" * 70)

    for ds_name, k, dimensions in datasets_config:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name} (k={k})")
        print(f"{'='*60}")

        outfile = os.path.join(results_dir, f"{METHOD}-{ds_name}-{N_SAMPLES}.txt")

        with open(outfile, "w") as f:
            # Write header
            f.write("Dataset\tADMM_InputGroup_bestK\tADMM_InputGroup_best2K\t"
                    "ADMM_InputGroup_TrainAUC\tADMM_InputGroup_TrainAUPRC\t"
                    "ADMM_InputGroup_AUC\tADMM_InputGroup_AUPRC\n")

            for n_features in dimensions:
                print(f"\n  m={n_features}:")

                # Set random seeds for reproducibility
                np.random.seed(SEED)
                torch.manual_seed(SEED)

                # Generate dataset using official benchmark's data generator
                X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, n_features)

                # Scale data to [-1, 1] (matching benchmark protocol)
                X = 2.0 * X - 1.0
                X_tilde = 2.0 * X_tilde - 1.0

                # 6-fold cross-validation
                splits = list(KFold(n_splits=6).split(X))

                t0 = time.time()
                best_ks = []
                best_2ks = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    best_k, best_2k = evaluate_fold(
                        train_idx, test_idx, X, X_tilde, y, k, n_features
                    )
                    best_ks.append(best_k)
                    best_2ks.append(best_2k)
                    print(f"    Fold {fold_idx+1}: best-k={best_k:.1%}, best-2k={best_2k:.1%}")

                elapsed = time.time() - t0

                # Aggregate results
                bk = np.mean(best_ks) if best_ks else 0
                b2k = np.mean(best_2ks) if best_2ks else 0

                print(f"    Average: best-k={bk:.1%}, best-2k={b2k:.1%} ({elapsed:.0f}s)")

                # Write to file
                row_name = f"{ds_name}_{n_features}_{N_SAMPLES}"
                f.write(f"{row_name}\t{bk}\t{b2k}\t\t\t\t\n")

        print(f"  Saved: {outfile}")

    print("\n" + "=" * 70)
    print("  Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
