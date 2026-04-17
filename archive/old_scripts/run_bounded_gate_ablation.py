"""Run bounded-vs-unbounded gate ablation for the appendix table.

Tasks:
  - XOR at m=512
  - Ring at m=512
  - Ring+XOR+Sum at m=512

Protocol:
  - 3 dataset seeds
  - 6-fold CV per seed
  - report mean/std over the 3 seed-level best-k means
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.admm_input_group_wrapper import run_admm_input_group
from src.data import generate_dataset

RESULTS_DIR = os.path.join(ROOT, "results", "ablations")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SAMPLES = 1000
N_FOLDS = 6
SEEDS = [0, 1, 2]
TASKS = [
    ("xor", 2, 512),
    ("ring", 2, 512),
    ("ring+xor+sum", 6, 512),
]


def run_one_seed(ds_name, k_true, m, bounded_gate, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X, _, y = generate_dataset(ds_name, N_SAMPLES, m)
    X = 2.0 * X - 1.0

    splitter = KFold(n_splits=N_FOLDS)
    best_ks = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        rng = np.random.RandomState(seed * 100 + fold_idx)
        perm = rng.permutation(m)
        X_train = X_train[:, perm]
        X_test = X_test[:, perm]
        correct = set(np.where(perm < k_true)[0].tolist())

        _, _, scores, _ = run_admm_input_group(
            X_train,
            y_train,
            X_test,
            n_classes=2,
            bounded_gate=bounded_gate,
            seed=seed * 100 + fold_idx,
        )
        ranked = np.argsort(np.abs(scores))
        best_ks.append(sum(i in correct for i in ranked[-k_true:]) / k_true)

    return float(np.mean(best_ks)), float(np.std(best_ks))


def summarize(vals):
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def main():
    results = {}
    for gate_name, bounded_gate in [("unbounded", False), ("bounded", True)]:
        gate_results = {}
        print(f"\n=== {gate_name} ===")
        for ds_name, k_true, m in TASKS:
            per_seed = []
            print(f"Running {ds_name} m={m} ...")
            for seed in SEEDS:
                bk_mean, bk_std = run_one_seed(ds_name, k_true, m, bounded_gate, seed)
                per_seed.append({"seed": seed, "best_k": bk_mean, "fold_std": bk_std})
                print(f"  seed={seed}: best-k={bk_mean:.1%} +/- {bk_std:.1%}")

            mean_bk, std_bk = summarize([x["best_k"] for x in per_seed])
            gate_results[f"{ds_name}_m{m}"] = {
                "m": m,
                "best_k": mean_bk,
                "best_k_std": std_bk,
                "per_seed": per_seed,
            }
            print(f"  summary: {mean_bk:.1%} +/- {std_bk:.1%}")
        results[gate_name] = gate_results

    out_path = os.path.join(RESULTS_DIR, "bounded_gate_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
