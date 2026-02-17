#!/usr/bin/env python
"""
Quick test: ADMM on xor & ring with warm_start=False.
Compares against warm_start=True to isolate the effect.
"""
import sys, os, time
import numpy as np
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(ROOT, "Feature-Selection-Benchmark")
sys.path.insert(0, BENCH)
sys.path.insert(0, ROOT)

from src.data import generate_dataset
from src.admm_lasso_wrapper import (
    FeatureSelectionMLP, _Scaler, _warm_start_from_lasso,
    _train_with_optimizer, _extract_feature_importance, _predict_proba,
    _OPTIMIZER_MAP,
)

METHODS = ["admm_global", "admm_layer", "admm_neuron"]
DATASETS = {
    "xor":  {"k": 2, "ns": [8, 32, 128]},
    "ring": {"k": 2, "ns": [8, 32, 128]},
}
N_SAMPLES = 500
N_FOLDS   = 3
SEED      = 42

LR, C_SPARSE, EPOCHS = 0.005, 0.08, 100


def run_one(method, dataset_name, k, n_features, warm_start):
    np.random.seed(SEED)
    import torch; torch.manual_seed(SEED)

    X, X_tilde, y = generate_dataset(dataset_name, N_SAMPLES, n_features)
    X = 2.0 * X - 1.0

    splits = list(KFold(n_splits=N_FOLDS, shuffle=False).split(X))
    best_ks = []

    opt_cls, is_admm = _OPTIMIZER_MAP[method]

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # permute features
        perm = np.arange(n_features)
        np.random.shuffle(perm)
        X_tr, X_te = X_tr[:, perm], X_te[:, perm]
        gt = set(np.where(perm < k)[0].tolist())

        scaler = _Scaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        n_classes = len(np.unique(y_tr))
        model = FeatureSelectionMLP(
            input_size=n_features, n_classes=n_classes,
            latent_size=58, n_hidden_layers=5,
            gaussian_noise=0.0, dropout=0.0, activation="mish",
        )

        if warm_start:
            _warm_start_from_lasso(model, X_tr_s, y_tr, n_classes)

        _train_with_optimizer(
            model, X_tr_s, y_tr, n_classes, opt_cls, is_admm,
            lr=LR, C=C_SPARSE, epochs=EPOCHS,
        )

        scores = _extract_feature_importance(model, X_tr_s)
        top_k = set(np.argsort(np.abs(scores))[-k:].tolist())
        best_ks.append(len(top_k & gt) / k)

    return np.mean(best_ks)


def main():
    print(f"{'method':<16} {'dataset':<8} {'n':>4}  {'ws=OFF':>8}  {'ws=ON':>8}")
    print("-" * 56)
    t0 = time.time()
    for method in METHODS:
        for ds, info in DATASETS.items():
            k = info["k"]
            for n in info["ns"]:
                bk_off = run_one(method, ds, k, n, warm_start=False)
                bk_on  = run_one(method, ds, k, n, warm_start=True)
                print(f"{method:<16} {ds:<8} {n:>4}  {bk_off:>8.2%}  {bk_on:>8.2%}")
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
