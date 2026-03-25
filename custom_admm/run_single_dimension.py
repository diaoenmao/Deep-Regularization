#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a single dimension of the benchmark.
Called by run_parallel_benchmark.py
"""
import sys
import os
import argparse
import time
import numpy as np
import torch
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data import generate_dataset
from admm_input_group_wrapper import run_admm_input_group

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n-features", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    n_features = args.n_features
    k = args.k
    ds_name = args.dataset

    # Generate dataset
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    X, X_tilde, y = generate_dataset(ds_name, args.n_samples, n_features)
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    splits = list(KFold(n_splits=6).split(X))
    best_ks = []
    best_2ks = []

    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
        y_train = y[train_idx]

        # Shuffle features
        idx = np.arange(n_features)
        np.random.shuffle(idx)
        X_train, X_test = X_train[:, idx], X_test[:, idx]
        correct = set(np.where(idx < k)[0].tolist())

        # Run ADMM
        _, _, scores, scores2 = run_admm_input_group(
            X_train, y_train, X_test, n_classes=2,
            use_ratio_norm=True, use_admm=True, seed=args.seed
        )

        # Compute metrics
        if scores is not None:
            ranked = np.argsort(np.abs(scores))
            best_k = sum(i in correct for i in ranked[-k:]) / k
        else:
            best_k = 0

        if scores2 is not None:
            ranked2 = np.argsort(np.abs(scores2))
            best_2k = sum(i in correct for i in ranked2[-(2*k):]) / k
        else:
            best_2k = 0

        best_ks.append(best_k)
        best_2ks.append(best_2k)

    elapsed = time.time() - t0
    bk = np.mean(best_ks) if best_ks else 0
    b2k = np.mean(best_2ks) if best_2ks else 0

    print(f"  Fold avg: best-k={bk:.1%}, best-2k={b2k:.1%} ({elapsed:.0f}s)")

    # Write result
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f"admm_input_group-{ds_name}-{args.n_samples}.txt")

    # Check if file exists, write header if not
    write_header = not os.path.exists(outfile)
    with open(outfile, "a") as f:
        if write_header:
            f.write("Dataset\tADMM_InputGroup_bestK\tADMM_InputGroup_best2K\n")
        row_name = f"{ds_name}_{n_features}_{args.n_samples}"
        f.write(f"{row_name}\t{bk}\t{b2k}\n")

    print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()
