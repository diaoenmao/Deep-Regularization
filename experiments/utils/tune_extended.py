#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended hyperparameter tuning for ADMM input_group method.

Phase 2: Extended Ablation Study

After Phase 1 (feat_drop ablation), this script tunes additional
hyperparameters to match original performance:

1. warmup_epochs: More warmup may help the MLP learn before pruning
2. C (sparsity coefficient): Controls pruning strength
3. lr (learning rate): Affects convergence during ADMM phase

Key insight from Phase 1:
- feat_drop=0.6 works better than 0.7 (original default)
- Performance drops for m > 64, suggesting need for more training

Test configuration:
- Dataset: ring (k=2)
- Feature dimensions: m = [64, 128] (most informative range)
- feat_drop: 0.6 (from Phase 1)
- warmup_epochs: [120, 200, 300]
- C: [0.01, 0.05, 0.1]
- lr: [0.005, 0.01]
"""

import sys
import os
import json
import time
import numpy as np
import torch
from sklearn.model_selection import KFold

# Add custom_admm/src to path
CUSTOM_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, CUSTOM_SRC)

from src.admm_input_group_wrapper import run_admm_input_group
from src.data import generate_dataset

# Configuration
SEED = 0
N_SAMPLES = 1000
DATASET = "ring"
K = 2
CV_FOLDS = 2

# Fixed from Phase 1
FEAT_DROP = 0.6

# Grid search parameters
WARMUP_EPOCHS = [120, 200, 300]
C_VALUES = [0.01, 0.05, 0.1]
LR_VALUES = [0.005, 0.01]

FEATURE_DIMS = [64, 128]  # Focus on most informative range

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tuning")
os.makedirs(results_dir, exist_ok=True)


def compute_best_k_scores(scores, true_informative_indices, k):
    """Compute best_k and best_2k scores."""
    ranked = np.argsort(np.abs(scores))[::-1]
    top_k = ranked[:k]
    best_k = sum(idx in true_informative_indices for idx in top_k) / k
    top_2k = ranked[:2*k]
    best_2k = sum(idx in true_informative_indices for idx in top_2k) / k
    return best_k, best_2k


def run_single_config(n_features, hp_overrides, verbose=False):
    """Run a single hyperparameter configuration."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, X_tilde, y = generate_dataset(DATASET, N_SAMPLES, n_features)
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    splits = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(X))
    best_ks = []
    best_2ks = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        idx = np.arange(n_features)
        np.random.shuffle(idx)
        X_train, X_test = X_train[:, idx], X_test[:, idx]
        true_informative_indices = set(np.where(idx < K)[0].tolist())

        try:
            _, _, scores, _ = run_admm_input_group(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                n_classes=2,
                seed=SEED + fold_idx,
                hp_overrides=hp_overrides,
            )
            if scores is not None:
                best_k, best_2k = compute_best_k_scores(scores, true_informative_indices, K)
                best_ks.append(best_k)
                best_2ks.append(best_2k)
        except Exception as e:
            if verbose:
                print(f"    Fold {fold_idx} failed: {e}")
            continue

    return {
        "best_k": np.mean(best_ks) if best_ks else 0.0,
        "best_2k": np.mean(best_2ks) if best_2ks else 0.0,
        "per_fold_bestk": best_ks,
        "success": len(best_ks) == CV_FOLDS,
    }


def main():
    """Run extended hyperparameter tuning."""
    print("=" * 70)
    print("Extended Hyperparameter Tuning (Phase 2)")
    print("=" * 70)
    print("")
    print(f"Dataset: {DATASET} (k={K})")
    print(f"Fixed feat_drop: {FEAT_DROP}")
    print(f"Feature dimensions: {FEATURE_DIMS}")
    print("")
    print("Grid search:")
    print(f"  warmup_epochs: {WARMUP_EPOCHS}")
    print(f"  C: {C_VALUES}")
    print(f"  lr: {LR_VALUES}")
    print("")
    print(f"Total configurations: {len(WARMUP_EPOCHS) * len(C_VALUES) * len(LR_VALUES)}")
    print("")

    all_results = []
    best_config = None
    best_avg_k = -1

    for warmup in WARMUP_EPOCHS:
        for c_val in C_VALUES:
            for lr_val in LR_VALUES:
                hp = {
                    "feat_drop": FEAT_DROP,
                    "warmup_epochs": warmup,
                    "C": c_val,
                    "lr": lr_val,
                    "epochs": 500,
                    "rho_init": 200.0,
                }

                print(f"\nTesting: warmup={warmup}, C={c_val}, lr={lr_val}")
                print("-" * 50)

                config_results = {
                    "warmup_epochs": warmup,
                    "C": c_val,
                    "lr": lr_val,
                    "per_m": [],
                    "avg_best_k": 0.0,
                }

                m_results = []
                for n_features in FEATURE_DIMS:
                    t0 = time.time()
                    result = run_single_config(n_features, hp)
                    elapsed = time.time() - t0

                    m_results.append({
                        "m": n_features,
                        "best_k": result["best_k"],
                        "success": result["success"],
                    })

                    status = "OK" if result["success"] else "PARTIAL"
                    print(f"  m={n_features:4d}: best_k={result['best_k']:.4f} ({status}, {elapsed:.1f}s)")

                config_results["per_m"] = m_results
                config_results["avg_best_k"] = np.mean([r["best_k"] for r in m_results])

                print(f"  --> avg_best_k = {config_results['avg_best_k']:.4f}")

                all_results.append(config_results)

                if config_results["avg_best_k"] > best_avg_k:
                    best_avg_k = config_results["avg_best_k"]
                    best_config = config_results.copy()

    # Summary
    print(f"\n{'='*70}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*70}")
    print("")
    print(f"{'warmup':>8} | {'C':>6} | {'lr':>6} | {'avg_best_k':>12} | {'m=64':>8} | {'m=128':>8}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}")

    # Sort by avg_best_k descending
    sorted_results = sorted(all_results, key=lambda x: x["avg_best_k"], reverse=True)

    for i, r in enumerate(sorted_results[:10]):
        m64_k = r["per_m"][0]["best_k"] if len(r["per_m"]) > 0 else 0
        m128_k = r["per_m"][1]["best_k"] if len(r["per_m"]) > 1 else 0
        marker = " <-- BEST" if i == 0 else ""
        print(f"{r['warmup_epochs']:>8} | {r['C']:>6.2f} | {r['lr']:>6.3f} | "
              f"{r['avg_best_k']:>12.4f} | {m64_k:>8.4f} | {m128_k:>8.4f}{marker}")

    print(f"\nBest configuration:")
    print(f"  warmup_epochs = {best_config['warmup_epochs']}")
    print(f"  C = {best_config['C']}")
    print(f"  lr = {best_config['lr']}")
    print(f"  feat_drop = {FEAT_DROP}")
    print(f"  avg_best_k = {best_config['avg_best_k']:.4f}")

    # Reference performance
    REFERENCE_RING_AVG_BEST_K = 0.6388888888888888  # Original results
    CURRENT_RING_AVG_BEST_K = 0.12037037037037036   # Consistent arch default

    print(f"\nComparison with reference:")
    print(f"  Original (backup):     {REFERENCE_RING_AVG_BEST_K:.4f}")
    print(f"  Consistent (default):  {CURRENT_RING_AVG_BEST_K:.4f}")
    print(f"  Tuned (this study):    {best_config['avg_best_k']:.4f}")
    print(f"  Improvement:           {best_config['avg_best_k'] - CURRENT_RING_AVG_BEST_K:+.4f}")
    print(f"  Gap to original:       {REFERENCE_RING_AVG_BEST_K - best_config['avg_best_k']:.4f}")

    # Save results
    output = {
        "dataset": DATASET,
        "n_samples": N_SAMPLES,
        "cv_folds": CV_FOLDS,
        "fixed_feat_drop": FEAT_DROP,
        "grid_search": {
            "warmup_epochs": WARMUP_EPOCHS,
            "C": C_VALUES,
            "lr": LR_VALUES,
        },
        "feature_dims_tested": FEATURE_DIMS,
        "all_results": all_results,
        "best_configuration": {
            "warmup_epochs": best_config["warmup_epochs"],
            "C": best_config["C"],
            "lr": best_config["lr"],
            "feat_drop": FEAT_DROP,
            "avg_best_k": best_config["avg_best_k"],
        },
        "reference_comparison": {
            "original_backup": REFERENCE_RING_AVG_BEST_K,
            "consistent_default": CURRENT_RING_AVG_BEST_K,
            "tuned": best_config["avg_best_k"],
        },
    }

    output_file = os.path.join(results_dir, "extended_hyperparameter_tuning_ring.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == "__main__":
    main()
