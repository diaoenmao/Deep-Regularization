#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tune_dag.py — Hyperparameter tuning for admm_input_group on DAG dataset

Goal: Improve bestK (chain features) from current 7.8% while maintaining
      bestK2 advantage (currently 16.6%, rank #1).

Sweep grid (2-stage):
  Stage 1 (coarse): C × rho_init × epochs × warmup
  Stage 2 (fine):   Top-5 configs, more seeds

Usage:
    python tune_dag.py                     # full coarse sweep (2 seeds)
    python tune_dag.py --stage fine        # fine-tune top configs (5 seeds)
    python tune_dag.py --C 0.02 0.05      # custom C values
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

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.admm_lasso_wrapper import (
    GatedFeatureSelectionMLP,
    _train_input_group,
    _extract_feature_importance,
)
from src.dag import load_dag_dataset

RESULTS_DIR = os.path.join(ROOT, "results", "dag_tuning")
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_dag(scores, k, k2):
    """Compute bestK, best2K, bestK2, best2K2 metrics."""
    m = len(scores)
    ranking = np.argsort(scores)[::-1]  # descending

    # bestK: top-k hit rate on chain features (0..k-1)
    true_k = set(range(k))
    topk = set(ranking[:k].tolist())
    bestK = len(topk & true_k) / k

    # best2K: top-2k hit rate on chain features
    top2k = set(ranking[:2*k].tolist())
    best2K = len(top2k & true_k) / k

    # bestK2: top-k2 hit rate on chain+fork features (0..k2-1)
    true_k2 = set(range(k2))
    topk2 = set(ranking[:k2].tolist())
    bestK2 = len(topk2 & true_k2) / k2

    # best2K2
    top2k2 = set(ranking[:2*k2].tolist())
    best2K2 = len(top2k2 & true_k2) / k2

    return {
        "bestK": bestK,
        "best2K": best2K,
        "bestK2": bestK2,
        "best2K2": best2K2,
    }


def run_dag_single(X, y, k, k2, seed, hp, feat_drop=0.7):
    """Train on DAG with given hyperparameters and return metrics."""
    from sklearn.preprocessing import StandardScaler

    n_features = X.shape[1]
    n_classes = int(np.max(y)) + 1

    # Shuffle
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X_s, y_s = X[idx], y[idx]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_s)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GatedFeatureSelectionMLP(
        input_size=n_features,
        n_classes=n_classes,
        latent_size=32,
        n_hidden_layers=2,
        feat_drop=feat_drop,
        activation="mish",
    )

    t0 = time.time()
    _train_input_group(
        model, X_s, y_s, n_classes,
        lr=hp["lr"],
        C=hp["C"],
        epochs=hp["epochs"],
        warmup_epochs=hp["warmup_epochs"],
        rho_init=hp["rho_init"],
    )
    elapsed = time.time() - t0

    scores = _extract_feature_importance(model, X_s)
    metrics = evaluate_dag(scores, k, k2)
    metrics["time"] = elapsed
    metrics["n_zero"] = int(np.sum(scores < 1e-6))
    return metrics


# ──────────────────────────────────────────────────────────
# Sweep grids
# ──────────────────────────────────────────────────────────

COARSE_GRID = {
    "C":             [0.01, 0.02, 0.05, 0.1, 0.2],
    "rho_init":      [50, 100, 200, 500, 1000],
    "epochs":        [500, 800, 1200],
    "warmup_epochs": [120, 200, 300],
    "lr":            [0.005],
}

COARSE_SEEDS = [0, 42]

FINE_SEEDS = [0, 42, 123, 456, 789]


def build_grid(grid_dict):
    """Generate list of dicts from grid specification."""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def main():
    parser = argparse.ArgumentParser(description="DAG hyperparameter tuning")
    parser.add_argument("--stage", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--C", nargs="+", type=float, default=None)
    parser.add_argument("--rho-init", nargs="+", type=float, default=None)
    parser.add_argument("--epochs", nargs="+", type=int, default=None)
    parser.add_argument("--warmup", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=5, help="Fine-tune top N configs")
    parser.add_argument("--feat-drop", type=float, default=0.7)
    args = parser.parse_args()

    # Load DAG data once
    print("Loading DAG dataset...")
    X, X_tilde, y, k, k2 = load_dag_dataset(os.path.join(ROOT, "data"))
    print(f"  X: {X.shape}, k={k}, k2={k2}, classes={int(np.max(y))+1}")

    if args.stage == "coarse":
        grid = dict(COARSE_GRID)
        if args.C:
            grid["C"] = args.C
        if args.rho_init:
            grid["rho_init"] = args.rho_init
        if args.epochs:
            grid["epochs"] = args.epochs
        if args.warmup:
            grid["warmup_epochs"] = args.warmup

        configs = build_grid(grid)
        seeds = args.seeds or COARSE_SEEDS
        total = len(configs) * len(seeds)
        print(f"\nCoarse sweep: {len(configs)} configs × {len(seeds)} seeds = {total} runs")

        results = []
        for i, (hp, seed) in enumerate(itertools.product(configs, seeds)):
            tag = f"[{i+1}/{total}] C={hp['C']:.3f} rho={hp['rho_init']:.0f} ep={hp['epochs']} wu={hp['warmup_epochs']} seed={seed}"
            print(tag, end=" ... ", flush=True)

            try:
                metrics = run_dag_single(X, y, k, k2, seed, hp, args.feat_drop)
                print(f"bestK={metrics['bestK']:.3f} bestK2={metrics['bestK2']:.3f} "
                      f"best2K2={metrics['best2K2']:.3f} {metrics['time']:.0f}s")
            except Exception as e:
                print(f"FAILED: {e}")
                metrics = {"bestK": 0, "bestK2": 0, "best2K2": 0, "error": str(e)}

            record = {**hp, "seed": seed, "feat_drop": args.feat_drop, **metrics}
            results.append(record)

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RESULTS_DIR, f"dag_coarse_{timestamp}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

        # Find top configs by average combined score
        print_top_configs(results, configs, seeds, args.top_n)

    elif args.stage == "fine":
        # Load coarse results to find top configs
        latest = find_latest_coarse()
        if latest is None:
            print("No coarse sweep results found. Run --stage coarse first.")
            sys.exit(1)

        with open(latest) as f:
            coarse_results = json.load(f)
        print(f"Loaded coarse results from {latest}")

        # Find top configs
        top_configs = get_top_configs(coarse_results, args.top_n)
        seeds = args.seeds or FINE_SEEDS
        total = len(top_configs) * len(seeds)
        print(f"\nFine-tuning top {len(top_configs)} configs × {len(seeds)} seeds = {total} runs")

        results = []
        for i, (hp, seed) in enumerate(itertools.product(top_configs, seeds)):
            tag = f"[{i+1}/{total}] C={hp['C']:.3f} rho={hp['rho_init']:.0f} ep={hp['epochs']} wu={hp['warmup_epochs']} seed={seed}"
            print(tag, end=" ... ", flush=True)

            try:
                metrics = run_dag_single(X, y, k, k2, seed, hp, args.feat_drop)
                print(f"bestK={metrics['bestK']:.3f} bestK2={metrics['bestK2']:.3f} "
                      f"best2K2={metrics['best2K2']:.3f} {metrics['time']:.0f}s")
            except Exception as e:
                print(f"FAILED: {e}")
                metrics = {"bestK": 0, "bestK2": 0, "best2K2": 0, "error": str(e)}

            record = {**hp, "seed": seed, "feat_drop": args.feat_drop, **metrics}
            results.append(record)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RESULTS_DIR, f"dag_fine_{timestamp}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
        print_fine_summary(results, top_configs, seeds)


def find_latest_coarse():
    """Find most recent coarse sweep file."""
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("dag_coarse_")]
    if not files:
        return None
    files.sort()
    return os.path.join(RESULTS_DIR, files[-1])


def get_top_configs(results, top_n):
    """Extract top N configs by combined bestK + bestK2 average across seeds."""
    from collections import defaultdict
    scores = defaultdict(list)
    for r in results:
        key = (r["C"], r["rho_init"], r["epochs"], r["warmup_epochs"], r["lr"])
        # Combined metric: weight bestK more since that's our weak point
        combined = 0.7 * r.get("bestK", 0) + 0.3 * r.get("bestK2", 0)
        scores[key].append(combined)

    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    top_keys = sorted(avg_scores, key=avg_scores.get, reverse=True)[:top_n]

    configs = []
    for k in top_keys:
        configs.append({
            "C": k[0], "rho_init": k[1], "epochs": k[2],
            "warmup_epochs": k[3], "lr": k[4],
        })
        print(f"  Top config: C={k[0]:.3f} rho={k[1]:.0f} ep={k[2]} wu={k[3]} "
              f"avg_combined={avg_scores[k]:.4f}")
    return configs


def print_top_configs(results, configs, seeds, top_n):
    """Print top configs from coarse sweep."""
    from collections import defaultdict
    avg = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = f"C={r['C']:.3f}_rho={r['rho_init']:.0f}_ep={r['epochs']}_wu={r['warmup_epochs']}"
        avg[key]["bestK"].append(r.get("bestK", 0))
        avg[key]["bestK2"].append(r.get("bestK2", 0))
        avg[key]["best2K2"].append(r.get("best2K2", 0))

    print(f"\n{'='*80}")
    print(f"TOP {top_n} CONFIGS (by 0.7*bestK + 0.3*bestK2)")
    print(f"{'='*80}")

    ranked = []
    for key, vals in avg.items():
        bk = np.mean(vals["bestK"])
        bk2 = np.mean(vals["bestK2"])
        b2k2 = np.mean(vals["best2K2"])
        combined = 0.7 * bk + 0.3 * bk2
        ranked.append((combined, key, bk, bk2, b2k2))

    ranked.sort(reverse=True)
    for i, (comb, key, bk, bk2, b2k2) in enumerate(ranked[:top_n]):
        print(f"  #{i+1}: {key}")
        print(f"       bestK={bk*100:.1f}%  bestK2={bk2*100:.1f}%  best2K2={b2k2*100:.1f}%  combined={comb:.4f}")

    # Baseline
    print(f"\n  Baseline (C=0.05 rho=200 ep=500 wu=120):")
    print(f"       bestK=7.8%  bestK2=16.6%  best2K2=32.4%")


def print_fine_summary(results, configs, seeds):
    """Print fine-tuning summary."""
    from collections import defaultdict
    avg = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = f"C={r['C']:.3f}_rho={r['rho_init']:.0f}_ep={r['epochs']}_wu={r['warmup_epochs']}"
        avg[key]["bestK"].append(r.get("bestK", 0))
        avg[key]["bestK2"].append(r.get("bestK2", 0))
        avg[key]["best2K2"].append(r.get("best2K2", 0))

    print(f"\n{'='*80}")
    print(f"FINE-TUNING RESULTS ({len(seeds)} seeds)")
    print(f"{'='*80}")

    for key, vals in avg.items():
        bk = np.mean(vals["bestK"])
        bk_std = np.std(vals["bestK"])
        bk2 = np.mean(vals["bestK2"])
        b2k2 = np.mean(vals["best2K2"])
        print(f"  {key}")
        print(f"    bestK={bk*100:.1f}%+/-{bk_std*100:.1f}%  "
              f"bestK2={bk2*100:.1f}%  best2K2={b2k2*100:.1f}%")


if __name__ == "__main__":
    main()
