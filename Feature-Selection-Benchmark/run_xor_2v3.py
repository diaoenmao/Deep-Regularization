# -*- coding: utf-8 -*-
"""Quick XOR comparison: 2-variable vs 3-variable ADMM.
Uses the benchmark's exact data generation and best-k metric.
"""
import sys, os, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '..'))

from src.admm_lasso_wrapper import (
    GatedFeatureSelectionMLP, _train_input_group, _extract_feature_importance,
)

N_SEEDS = 3

def generate_xor(m=128, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, m))
    y = ((X[:, 0] - 0.5) * (0.5 - X[:, 1]) >= 0).astype(int)
    X = 2 * X - 1  # center to [-1, 1]
    return X, y

def evaluate_bestk(scores, k):
    top_idx = set(np.argsort(scores)[-k:])
    true_set = set(range(k))
    return len(top_idx & true_set) / k

def run_xor(three_variable, seed, m=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = generate_xor(m=m, seed=seed)
    k = 2
    n_classes = 2

    model = GatedFeatureSelectionMLP(
        input_size=m, n_classes=n_classes,
        latent_size=32, n_hidden_layers=2, feat_drop=0.7, activation="mish",
    )
    t0 = time.time()
    _train_input_group(
        model, X, y, n_classes,
        lr=0.005, C=0.05, epochs=500, warmup_epochs=120,
        rho_init=200.0, use_ratio_norm=True,
        three_variable=three_variable,
    )
    elapsed = time.time() - t0
    scores = _extract_feature_importance(model, X)
    bestk = evaluate_bestk(scores, k)
    return bestk, elapsed

if __name__ == '__main__':
    print("=" * 50, flush=True)
    print("XOR m=128: 2-variable vs 3-variable ADMM", flush=True)
    print("=" * 50, flush=True)

    for label, three_var in [("2-variable (code)", False), ("3-variable (paper)", True)]:
        print(f"\n--- {label} ---", flush=True)
        bestks = []
        for s in range(N_SEEDS):
            bk, t = run_xor(three_var, seed=42 + s)
            bestks.append(bk)
            print(f"  Seed {s+1}: bestK={bk:.1%} ({t:.1f}s)", flush=True)
        print(f"  Mean: bestK={np.mean(bestks):.1%}±{np.std(bestks):.1%}", flush=True)
