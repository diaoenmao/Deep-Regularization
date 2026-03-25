# -*- coding: utf-8 -*-
"""
run_multiseed_dag.py — Multi-seed DAG evaluation across diverse graph instances.

Generates N random DAGs with different seeds and evaluates key methods on each,
reporting mean±std of bestK, bestK2, best2K, best2K2 across graphs.

Usage:
    python run_multiseed_dag.py                    # 5 graphs, all methods
    python run_multiseed_dag.py --n_graphs 3       # 3 graphs
    python run_multiseed_dag.py --methods admm_input_group stg rf
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dag import generate_dag_dataset
from src.core import run_fs_method
from src.admm_input_group_wrapper import (
    GatedFeatureSelectionMLP, _train_input_group, _extract_feature_importance,
)
from src.stg_wrapper import run_stg_fs

RESULTS_DIR = os.path.join(ROOT, "results", "dag_multiseed")
os.makedirs(RESULTS_DIR, exist_ok=True)

# DAG generation parameters (same as existing dag.npz)
DAG_PARAMS = dict(n_samples=1000, n_features=2000, min_n_relevant=20,
                  min_n_irrelevant=1000, density=0.004, sigma=0.2)

# ADMM hyperparameters (default from paper)
ADMM_HP = dict(lr=1e-3, C=0.05, epochs=500, warmup_epochs=120, rho_init=200)

METHODS = ['admm_input_group', 'stg', 'rf', 'treeshap', 'mi', 'relief', 'lassonet']


def evaluate_dag(scores, k, k2):
    """Compute bestK, best2K, bestK2, best2K2 metrics."""
    ranking = np.argsort(scores)[::-1]
    true_k = set(range(k))
    true_k2 = set(range(k2))

    topk = set(ranking[:k].tolist())
    top2k = set(ranking[:2 * k].tolist())
    topk2 = set(ranking[:k2].tolist())
    top2k2 = set(ranking[:2 * k2].tolist())

    return {
        "bestK": len(topk & true_k) / k,
        "best2K": len(top2k & true_k) / k,
        "bestK2": len(topk2 & true_k2) / k2,
        "best2K2": len(top2k2 & true_k2) / k2,
    }


def run_admm_on_dag(X, y, k, k2, seed):
    """Run admm_input_group directly (bypasses core.py for DAG-specific setup)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X_s, y_s = X[idx], y[idx]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_s)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GatedFeatureSelectionMLP(
        input_size=X.shape[1], n_classes=int(np.max(y)) + 1,
        latent_size=32, n_hidden_layers=2, feat_drop=0.7, activation="mish",
    )
    _train_input_group(model, X_s, y_s, int(np.max(y)) + 1, **ADMM_HP)
    scores = _extract_feature_importance(model, X_s)
    return evaluate_dag(scores, k, k2)


def run_stg_on_dag(X, y, k, k2, seed):
    """Run STG directly."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X_s, y_s = X[idx], y[idx]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_s)

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = int(np.max(y)) + 1
    _, _, scores, _ = run_stg_fs(X_s.astype(np.float32), y_s, X_s.astype(np.float32), n_classes)
    return evaluate_dag(scores, k, k2)


def run_classical_on_dag(X, y, k, k2, method, seed):
    """Run classical methods (rf, treeshap, mi, relief, lassonet) via core.py."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X_s, y_s = X[idx], y[idx]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_s).astype(np.float32)

    np.random.seed(seed)
    torch.manual_seed(seed)

    _, _, scores, _ = run_fs_method(
        'dag', method, X_s, X_s, y_s, X_s, X_s, k)
    return evaluate_dag(scores, k, k2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_graphs', type=int, default=5)
    parser.add_argument('--methods', nargs='+', default=METHODS)
    parser.add_argument('--method_seed', type=int, default=42,
                        help='Seed for method training (fixed across graphs)')
    args = parser.parse_args()

    # Pre-screened seeds that produce valid DAGs (k≥5)
    VALID_SEEDS = [1888, 2665, 3664, 4663, 5107]
    graph_seeds = VALID_SEEDS[:args.n_graphs]

    all_results = {}  # method -> list of metric dicts (one per graph)

    for gi, gseed in enumerate(graph_seeds):
        print(f"\n{'='*60}")
        print(f"Graph {gi+1}/{len(graph_seeds)} (seed={gseed})")
        print(f"{'='*60}")

        np.random.seed(gseed)
        X, y, k, k2 = generate_dag_dataset(**DAG_PARAMS)
        print(f"  k={k} (chain), k2={k2} (chain+fork), m={X.shape[1]}")

        for method in args.methods:
            print(f"  Running {method}...", end=" ", flush=True)
            t0 = time.time()

            try:
                if method == 'admm_input_group':
                    metrics = run_admm_on_dag(X, y, k, k2, args.method_seed)
                elif method == 'stg':
                    metrics = run_stg_on_dag(X, y, k, k2, args.method_seed)
                else:
                    metrics = run_classical_on_dag(X, y, k, k2, method, args.method_seed)

                elapsed = time.time() - t0
                metrics['time'] = elapsed
                metrics['graph_seed'] = gseed
                print(f"bestK={metrics['bestK']*100:.1f}% bestK2={metrics['bestK2']*100:.1f}% "
                      f"best2K2={metrics['best2K2']*100:.1f}% ({elapsed:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                metrics = {'bestK': 0, 'best2K': 0, 'bestK2': 0, 'best2K2': 0,
                           'time': 0, 'graph_seed': gseed, 'error': str(e)}

            all_results.setdefault(method, []).append(metrics)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.n_graphs} graphs)")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'bestK':>10} {'bestK2':>10} {'best2K2':>10}")
    print("-" * 55)

    summary = {}
    for method in args.methods:
        results = all_results.get(method, [])
        if not results:
            continue
        bk = [r['bestK'] for r in results]
        bk2 = [r['bestK2'] for r in results]
        b2k2 = [r['best2K2'] for r in results]
        summary[method] = {
            'bestK_mean': float(np.mean(bk)), 'bestK_std': float(np.std(bk)),
            'bestK2_mean': float(np.mean(bk2)), 'bestK2_std': float(np.std(bk2)),
            'best2K2_mean': float(np.mean(b2k2)), 'best2K2_std': float(np.std(b2k2)),
            'per_graph': results,
        }
        print(f"{method:<20} {np.mean(bk)*100:>5.1f}±{np.std(bk)*100:>4.1f}% "
              f"{np.mean(bk2)*100:>5.1f}±{np.std(bk2)*100:>4.1f}% "
              f"{np.mean(b2k2)*100:>5.1f}±{np.std(b2k2)*100:>4.1f}%")

    out_path = os.path.join(RESULTS_DIR, f"multiseed_dag_{args.n_graphs}graphs.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
