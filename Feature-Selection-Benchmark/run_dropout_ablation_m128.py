"""
Run dropout ablation for the paper table:
  - XOR at m=128 and m=256 (m=128 is too easy, m=256 shows dropout effect)
  - Ring at m=128
Dropout rates: [0.0, 0.3, 0.5, 0.7]
"""
import sys, os, json
import numpy as np
import torch
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data import generate_dataset
from src.admm_lasso_wrapper import run_admm_lasso_fs

SEED = 0
N_SAMPLES = 1000
RESULTS_PATH = os.path.join(ROOT, "results", "ablations")
os.makedirs(RESULTS_PATH, exist_ok=True)


def run_single(ds_name, k_true, m, feat_drop):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, m)
    X = 2.0 * X - 1.0

    splits = list(KFold(n_splits=6).split(X))
    best_ks = []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        idx = np.arange(m)
        np.random.shuffle(idx)
        X_train, X_test = X_train[:, idx], X_test[:, idx]
        correct = set(np.where(idx < k_true)[0].tolist())

        _, _, scores, _ = run_admm_lasso_fs(
            "admm_input_group", X_train, y_train, X_test, n_classes=2,
            hp_overrides={"feat_drop": feat_drop}
        )
        if scores is not None:
            ranked = np.argsort(np.abs(scores))
            best_ks.append(sum(i in correct for i in ranked[-k_true:]) / k_true)

    bk = np.mean(best_ks) if best_ks else 0
    bk_std = np.std(best_ks) if best_ks else 0
    return bk, bk_std


if __name__ == "__main__":
    dropout_rates = [0.0, 0.3, 0.5, 0.7]
    tasks = [
        ("xor", 2, 128),
        ("xor", 2, 256),
        ("ring", 2, 128),
    ]

    results = {}
    for ds_name, k_true, m in tasks:
        key = f"{ds_name}_m{m}"
        results[key] = {}
        for dr in dropout_rates:
            print(f"Running {ds_name} m={m} dropout={dr}...")
            bk, bk_std = run_single(ds_name, k_true, m, dr)
            results[key][str(dr)] = {
                "m": m, "best_k": bk, "best_k_std": bk_std
            }
            print(f"  best-k={bk:.1%} +/- {bk_std:.1%}")

    out_path = os.path.join(RESULTS_PATH, "dropout_ablation_m128_m256.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
