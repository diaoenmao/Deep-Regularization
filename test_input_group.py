#!/usr/bin/env python
"""
Test: GatedMLP + Feature Dropout for nonlinear feature selection.
Uses the integrated wrapper (run_admm_lasso_fs) for admm_input_group.
Compares against admm_global and rf baselines.
"""
import sys, os, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(ROOT, "Feature-Selection-Benchmark")
sys.path.insert(0, BENCH)
sys.path.insert(0, ROOT)

from src.data import generate_dataset
from src.admm_lasso_wrapper import run_admm_lasso_fs

METHODS = ["admm_global", "admm_input_group"]
DATASETS = {
    "xor":          {"gt": [0, 1]},
    "ring":         {"gt": [0, 1]},
    "ring+xor+sum": {"gt": [0, 1, 2, 3, 4, 5]},
}
N_FEATURES = 128
N_SAMPLES  = 1000
N_SEEDS    = 3


def run_one(method, dataset_name, gt, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X, _, y = generate_dataset(dataset_name, N_SAMPLES, N_FEATURES)
    n_tr = int(0.7 * len(y))
    idx = np.random.permutation(len(y))
    X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
    y_tr = y[idx[:n_tr]]
    nc = len(np.unique(y_tr))

    _, _, scores, _ = run_admm_lasso_fs(method, X_tr, y_tr, X_te, nc)
    k = len(gt)
    top_k = set(np.argsort(scores)[::-1][:k].tolist())
    bk = len(top_k & set(gt)) / k
    nz = int(np.sum(np.abs(scores) < 1e-6))
    return bk, nz, sorted(top_k)


def main():
    print(f"{'method':<22} {'dataset':<15} {'mean-bk':>8}  {'per-seed':>30}  zeros")
    print("-" * 90)
    t0 = time.time()
    for method in METHODS:
        for ds, info in DATASETS.items():
            gt = info["gt"]
            bks, nzs = [], []
            for seed in range(N_SEEDS):
                bk, nz, top = run_one(method, ds, gt, seed)
                bks.append(bk)
                nzs.append(nz)
            bk_strs = " ".join(f"{b:.0%}" for b in bks)
            nz_str = "/".join(str(n) for n in nzs)
            print(f"{method:<22} {ds:<15} {np.mean(bks):>8.0%}  {bk_strs:>30}  {nz_str}")
        print()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
