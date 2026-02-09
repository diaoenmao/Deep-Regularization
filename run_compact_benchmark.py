#!/usr/bin/env python
"""
Compact benchmark test: run all 6 ADMM/Lasso methods on all 5 synthetic
datasets with moderate feature counts, then print a summary table.

Used to quickly validate whether the integration works and produces
reasonable results before committing to the full (slow) benchmark.
"""
import sys, os
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(ROOT, "Feature-Selection-Benchmark")
sys.path.insert(0, BENCH)
sys.path.insert(0, ROOT)

from src.data import generate_dataset
from src.core import run_fs_method


METHODS = [
    "admm_global", "admm_layer", "admm_neuron",
    "lasso_global", "lasso_layer", "lasso_neuron",
]

DATASETS = {
    "xor":          {"k": 2, "ns": [8, 32, 128]},
    "ring":         {"k": 2, "ns": [8, 32, 128]},
    "ring+xor":     {"k": 4, "ns": [8, 32, 128]},
    "ring+xor+sum": {"k": 6, "ns": [8, 32, 128]},
}

N_SAMPLES = 500   # quicker than 1000
N_FOLDS = 3       # quicker than 6


def run_one(method, dataset_name, k, n_features, n_samples, n_folds):
    """Run one method on one dataset/feature config with k-fold CV."""
    X, X_tilde, y = generate_dataset(dataset_name, n_samples, n_features)
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    splits = list(KFold(n_splits=n_folds).split(X))
    best_ks = []
    best_2ks = []
    aurocs = []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Randomly permute feature order (as the benchmark does)
        idx = np.arange(n_features)
        np.random.shuffle(idx)
        X_train, X_test = X_train[:, idx], X_test[:, idx]
        X_tilde_train, X_tilde_test = X_tilde_train[:, idx], X_tilde_test[:, idx]
        correct_indices = set(np.where(idx < k)[0].tolist())

        y_train_hat, y_hat, scores, scores2 = run_fs_method(
            dataset_name, method,
            X_train, X_tilde_train, y_train,
            X_test, X_tilde_test, k,
        )

        if scores is not None:
            top_k = np.argsort(np.abs(scores))[-k:]
            best_ks.append(sum(1 for i in top_k if i in correct_indices) / k)
            top_2k = np.argsort(np.abs(scores2))[-2*k:]
            best_2ks.append(sum(1 for i in top_2k if i in correct_indices) / k)

        if y_hat is not None:
            try:
                aurocs.append(roc_auc_score(y_test, y_hat))
            except:
                pass

    return {
        "best_k": np.mean(best_ks) if best_ks else float("nan"),
        "best_2k": np.mean(best_2ks) if best_2ks else float("nan"),
        "auroc": np.mean(aurocs) if aurocs else float("nan"),
    }


if __name__ == "__main__":
    np.random.seed(0xCAFE)
    results = {}

    total = sum(len(v["ns"]) for v in DATASETS.values()) * len(METHODS)
    done = 0
    t_start = time.time()

    for ds_name, ds_cfg in DATASETS.items():
        for n_feat in ds_cfg["ns"]:
            for method in METHODS:
                done += 1
                tag = f"[{done}/{total}]"
                print(f"{tag} {method:15s} | {ds_name:15s} | m={n_feat:4d}", end=" ", flush=True)
                t0 = time.time()
                r = run_one(method, ds_name, ds_cfg["k"], n_feat, N_SAMPLES, N_FOLDS)
                dt = time.time() - t0
                print(f"  bestK={r['best_k']:.2f}  best2K={r['best_2k']:.2f}  "
                      f"AUROC={r['auroc']:.3f}  ({dt:.1f}s)")
                results[(ds_name, n_feat, method)] = r

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"Completed in {elapsed:.0f}s")
    print(f"{'='*80}\n")

    # ---- Summary table ----
    print(f"{'Method':<16}", end="")
    for ds_name in DATASETS:
        for n_feat in DATASETS[ds_name]["ns"]:
            print(f" {ds_name[:6]}_{n_feat:>4}", end="")
    print("   AVG")

    for method in METHODS:
        print(f"{method:<16}", end="")
        vals = []
        for ds_name in DATASETS:
            for n_feat in DATASETS[ds_name]["ns"]:
                r = results[(ds_name, n_feat, method)]
                v = r["best_k"]
                vals.append(v)
                print(f"     {v:5.1%}", end="")
        avg = np.nanmean(vals)
        print(f"  {avg:5.1%}")

    print()
    print("Metric: best-k (fraction of true informative features found in top-k)")
