#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ablation.py — Ablation experiments for Linearized ADMM + Ratio Norm

Ablation grid:
  Sparsity method:  {ADMM+RatioNorm, ADMM+L1, ProxGrad+RatioNorm, ProxGrad+L1}
  Feature dropout:  {0.0, 0.3, 0.5, 0.7, 0.9}

Evaluation datasets:
  - xor     (m=128, k=2)   — interaction detection
  - ring+xor+sum (m=128, k=6) — mixed nonlinearity
  - dag     (m=2000)       — causal feature selection

3 seeds per config → 4 × 5 × 3 × 3 = 180 total runs

Usage:
    python run_ablation.py                # full grid
    python run_ablation.py --quick        # reduced grid (feat_drop={0.0, 0.7} only)
    python run_ablation.py --methods admm_ratio admm_l1
    python run_ablation.py --datasets xor dag
"""

import os
import sys
import json
import time
import argparse
import itertools
from datetime import datetime

import numpy as np
import torch

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.admm_lasso_wrapper import (
    GatedFeatureSelectionMLP,
    _train_input_group,
    _extract_feature_importance,
    _Scaler,
)
from src.dag import load_dag_dataset

RESULTS_DIR = os.path.join(ROOT, "results", "ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────
# Synthetic data generation (from main-benchmark.py)
# ──────────────────────────────────────────────────────────

def generate_synthetic(dataset_name: str, n_features: int, n_samples: int = 1000, seed: int = 42):
    """Generate a synthetic dataset and return (X, y, k)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_samples, n_features))

    if dataset_name == "xor":
        k = 2
        y = ((X[:, 0] - 0.5) * (0.5 - X[:, 1]) >= 0).astype(int)
    elif dataset_name == "ring":
        k = 2
        r = np.sqrt((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2)
        y = (np.abs(r - 0.35) <= 0.1151).astype(int)
    elif dataset_name == "ring+xor":
        k = 4
        r = np.sqrt((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2)
        ring = np.abs(r - 0.35) <= 0.10
        xor = (X[:, 2] - 0.5) * (0.5 - X[:, 3]) >= 0
        y = (ring | xor).astype(int)
    elif dataset_name == "ring+xor+sum":
        k = 6
        r = np.sqrt((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2)
        ring = np.abs(r - 0.35) <= 0.10
        xor = (X[:, 2] - 0.5) * (0.5 - X[:, 3]) >= 0
        noise = rng.normal(0, 0.1, n_samples)
        sumf = X[:, 4] + X[:, 5] + noise >= 1.41
        y = (ring | xor | sumf).astype(int)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Center to [-1, 1]
    X = 2 * X - 1
    return X, y, k


# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────

def evaluate_bestk(scores: np.ndarray, k: int, n_features: int) -> float:
    """Compute best-k: fraction of top-k scores that are true features (0..k-1)."""
    top_idx = np.argsort(scores)[-k:]
    true_set = set(range(k))
    hits = sum(1 for i in top_idx if i in true_set)
    return hits / k


def run_single(
    dataset_name: str,
    n_features: int,
    seed: int,
    feat_drop: float,
    use_ratio_norm: bool,
    use_admm: bool,
    hp_overrides: dict = None,
) -> dict:
    """Run one ablation experiment and return metrics."""
    hp = {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "rho_init": 200.0,
    }
    if hp_overrides:
        hp.update(hp_overrides)

    # ── Generate / load data ──
    if dataset_name == "dag":
        X, X_tilde, y, k, k2 = load_dag_dataset(os.path.join(ROOT, "data"))
        n_features_actual = X.shape[1]
        n_classes = int(np.max(y)) + 1
        # Shuffle with seed
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        # StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        # Synthetic: permute feature order to avoid position bias
        X, y, k = generate_synthetic(dataset_name, n_features, n_samples=1000, seed=seed)
        n_features_actual = n_features
        n_classes = int(np.max(y)) + 1
        rng = np.random.RandomState(seed + 1000)
        perm = rng.permutation(n_features)
        X = X[:, perm]
        # Remap truth indices
        inv_perm = np.argsort(perm)  # inv_perm[old_j] = new_position
        true_features = set(inv_perm[:k].tolist())

    # ── Build & train model ──
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GatedFeatureSelectionMLP(
        input_size=n_features_actual,
        n_classes=n_classes,
        latent_size=32,
        n_hidden_layers=2,
        feat_drop=feat_drop,
        activation="mish",
    )

    t0 = time.time()
    _train_input_group(
        model, X, y, n_classes,
        lr=hp["lr"], C=hp["C"], epochs=hp["epochs"],
        warmup_epochs=hp["warmup_epochs"],
        rho_init=hp["rho_init"],
        use_ratio_norm=use_ratio_norm,
        use_admm=use_admm,
    )
    elapsed = time.time() - t0

    # ── Evaluate ──
    scores = _extract_feature_importance(model, X)

    if dataset_name == "dag":
        bestK = evaluate_bestk(scores, k, n_features_actual)
        bestK2 = evaluate_bestk(scores, k2, n_features_actual) if k2 else None
        return {
            "bestK": bestK,
            "bestK2": bestK2,
            "time": elapsed,
            "n_zero": int(np.sum(scores < 1e-6)),
        }
    else:
        # After permutation, true features are at positions in true_features set
        top_idx = set(np.argsort(scores)[-k:].tolist())
        hits = len(top_idx & true_features)
        bestK = hits / k
        return {
            "bestK": bestK,
            "time": elapsed,
            "n_zero": int(np.sum(scores < 1e-6)),
        }


# ──────────────────────────────────────────────────────────
# Method variants
# ──────────────────────────────────────────────────────────

METHOD_VARIANTS = {
    "admm_ratio":    {"use_ratio_norm": True,  "use_admm": True,  "label": "ADMM + Ratio Norm"},
    "admm_l1":       {"use_ratio_norm": False, "use_admm": True,  "label": "ADMM + L1"},
    "prox_ratio":    {"use_ratio_norm": True,  "use_admm": False, "label": "ProxGrad + Ratio Norm"},
    "prox_l1":       {"use_ratio_norm": False, "use_admm": False, "label": "ProxGrad + L1"},
}

FEAT_DROP_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]
FEAT_DROP_QUICK = [0.0, 0.7]

DATASET_CONFIGS = {
    "xor":          {"n_features": 128},
    "ring+xor+sum": {"n_features": 128},
    "dag":          {"n_features": 2000},  # ignored, loaded from file
}

SEEDS = [0, 42, 123]


def main():
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--quick", action="store_true", help="Reduced grid (feat_drop={0.0, 0.7})")
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_VARIANTS.keys()),
                        default=list(METHOD_VARIANTS.keys()))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()),
                        default=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--feat-drops", nargs="+", type=float, default=None)
    args = parser.parse_args()

    feat_drops = args.feat_drops or (FEAT_DROP_QUICK if args.quick else FEAT_DROP_VALUES)

    # Build experiment grid
    grid = list(itertools.product(args.methods, feat_drops, args.datasets, args.seeds))
    total = len(grid)
    print(f"Ablation experiment: {total} runs")
    print(f"  Methods:    {args.methods}")
    print(f"  Feat drops: {feat_drops}")
    print(f"  Datasets:   {args.datasets}")
    print(f"  Seeds:      {args.seeds}")
    print()

    results = []
    t_start = time.time()

    for i, (method_key, fd, ds, seed) in enumerate(grid):
        mv = METHOD_VARIANTS[method_key]
        cfg = DATASET_CONFIGS[ds]
        tag = f"[{i+1}/{total}] {mv['label']:25s} fd={fd:.1f} {ds:15s} seed={seed}"
        print(tag, end=" ... ", flush=True)

        try:
            metrics = run_single(
                dataset_name=ds,
                n_features=cfg["n_features"],
                seed=seed,
                feat_drop=fd,
                use_ratio_norm=mv["use_ratio_norm"],
                use_admm=mv["use_admm"],
            )
            print(f"bestK={metrics['bestK']:.3f}  zeros={metrics['n_zero']}  {metrics['time']:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            metrics = {"bestK": None, "error": str(e)}

        record = {
            "method": method_key,
            "label": mv["label"],
            "use_ratio_norm": mv["use_ratio_norm"],
            "use_admm": mv["use_admm"],
            "feat_drop": fd,
            "dataset": ds,
            "seed": seed,
            "n_features": cfg["n_features"],
            **metrics,
        }
        results.append(record)

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total / 60:.1f} min")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"ablation_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Also save as latest
    latest_path = os.path.join(RESULTS_DIR, "ablation_latest.json")
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary table ──
    print_summary(results, args.methods, feat_drops, args.datasets)


def print_summary(results, methods, feat_drops, datasets):
    """Print a summary table of average best-k across seeds."""
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (average best-k % across seeds)")
    print("=" * 80)

    for ds in datasets:
        print(f"\n--- {ds} ---")
        header = f"{'Method':<28s}"
        for fd in feat_drops:
            header += f"  fd={fd:.1f}"
        print(header)
        print("-" * len(header))

        for mk in methods:
            mv = METHOD_VARIANTS[mk]
            row = f"{mv['label']:<28s}"
            for fd in feat_drops:
                vals = [r["bestK"] for r in results
                        if r["method"] == mk and r["feat_drop"] == fd
                        and r["dataset"] == ds and r["bestK"] is not None]
                if vals:
                    avg = np.mean(vals) * 100
                    row += f"  {avg:5.1f}%"
                else:
                    row += f"    N/A"
            print(row)


if __name__ == "__main__":
    main()
