# -*- coding: utf-8 -*-
"""Quick comparison: 2-variable ADMM (code) vs 3-variable ADMM (paper).

Runs on madelon (fast, real-world) and XOR m=128 (synthetic, interaction detection).
3 seeds each. Reports downstream AUROC (madelon) and best-k (XOR).
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

N_SEEDS = 3

# ── Load MADELON ──────────────────────────────────────────────────────
def load_madelon():
    data_dir = os.path.join(ROOT, 'data', 'madelon', 'MADELON')
    X_train = np.loadtxt(os.path.join(data_dir, 'madelon_train.data'))
    y_train = np.loadtxt(os.path.join(data_dir, 'madelon_train.labels'))
    X_test = np.loadtxt(os.path.join(data_dir, 'madelon_valid.data'))
    # valid labels are one level up
    y_test = np.loadtxt(os.path.join(ROOT, 'data', 'madelon', 'madelon_valid.labels'))
    y_train = ((y_train + 1) / 2).astype(int)
    y_test = ((y_test + 1) / 2).astype(int)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, y_train, X_test, y_test

# ── Load XOR synthetic ────────────────────────────────────────────────
def load_xor(m=128, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, m))
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)
    # shuffle feature order
    perm = rng.permutation(m)
    X = X[:, perm]
    true_features = set(np.where(perm < 2)[0])
    return X, y, true_features, 2  # k=2 informative features

# ── Run one experiment ────────────────────────────────────────────────
def run_madelon(three_variable, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, y_train, X_test, y_test = load_madelon()
    n_features = X_train.shape[1]
    n_classes = 2
    k = 25  # madelon has ~20 informative features

    model = GatedFeatureSelectionMLP(
        input_size=n_features, n_classes=n_classes,
        latent_size=32, n_hidden_layers=2, feat_drop=0.7, activation="mish",
    )
    t0 = time.time()
    _train_input_group(
        model, X_train, y_train, n_classes,
        lr=0.005, C=0.05, epochs=500, warmup_epochs=120,
        rho_init=200.0, use_ratio_norm=True,
        three_variable=three_variable,
    )
    elapsed = time.time() - t0

    scores = _extract_feature_importance(model, X_train)
    idx = np.argsort(scores)[-k:]
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    rf.fit(X_train[:, idx], y_train)
    y_prob = rf.predict_proba(X_test[:, idx])[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    return auroc, elapsed

def run_xor(three_variable, seed, m=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y, true_features, k = load_xor(m=m, seed=seed)
    n_features = X.shape[1]
    n_classes = 2

    model = GatedFeatureSelectionMLP(
        input_size=n_features, n_classes=n_classes,
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
    top_k = set(np.argsort(scores)[-k:])
    bestk = len(top_k & true_features) / k
    return bestk, elapsed

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("2-variable vs 3-variable ADMM comparison")
    print("=" * 60)

    for label, three_var in [("2-variable (code)", False), ("3-variable (paper)", True)]:
        print(f"\n--- {label} ---")

        # Madelon
        aurocs, times = [], []
        for s in range(N_SEEDS):
            auroc, t = run_madelon(three_var, seed=42 + s)
            aurocs.append(auroc)
            times.append(t)
            print(f"  MADELON seed {s+1}: AUROC={auroc:.4f} ({t:.1f}s)")
        print(f"  MADELON mean: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}")

        # XOR m=128
        bestks, times = [], []
        for s in range(N_SEEDS):
            bk, t = run_xor(three_var, seed=42 + s)
            bestks.append(bk)
            times.append(t)
            print(f"  XOR-128 seed {s+1}: bestK={bk:.1%} ({t:.1f}s)")
        print(f"  XOR-128 mean: bestK={np.mean(bestks):.1%}±{np.std(bestks):.1%}")
