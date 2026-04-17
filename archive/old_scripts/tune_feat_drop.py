#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameter tuning script for ADMM input_group method.

Phase 1: Feature Dropout Ablation Study

This script tests different feature dropout rates to find the optimal
configuration that matches the original ADMM performance.

Key hypothesis: The current feat_drop=0.7 (70% dropout) is too aggressive,
causing information loss and poor feature selection performance.

Test configuration:
- Dataset: ring (fastest to evaluate)
- Feature dimensions: m = [32, 64, 128, 256]
- Feature dropout: [0.0, 0.2, 0.4, 0.6]
- Cross-validation: 2-fold (for speed)
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
DATASET = "ring"  # Fastest dataset for tuning
K = 2  # Number of informative features for ring
CV_FOLDS = 2  # Use 2-fold for speed

# Hyperparameter grid
FEAT_DROP_VALUES = [0.0, 0.2, 0.4, 0.6]
FEATURE_DIMS = [32, 64, 128, 256]

# Other hyperparameters (fixed for this study)
HP_FIXED = {
    "lr": 0.005,
    "C": 0.05,
    "epochs": 500,
    "warmup_epochs": 120,
    "rho_init": 200.0,
}

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tuning")
os.makedirs(results_dir, exist_ok=True)


def compute_best_k_scores(scores, true_informative_indices, k):
    """Compute best_k and best_2k scores.

    Args:
        scores: Feature importance scores (array of length m)
        true_informative_indices: Set of indices of truly informative features
        k: Number of truly informative features

    Returns:
        best_k: Fraction of top-k selected features that are truly informative
        best_2k: Fraction of top-2k selected features that are truly informative
    """
    ranked = np.argsort(np.abs(scores))[::-1]  # Descending order

    # Top-k selected features
    top_k = ranked[:k]
    best_k = sum(idx in true_informative_indices for idx in top_k) / k

    # Top-2k selected features
    top_2k = ranked[:2*k]
    best_2k = sum(idx in true_informative_indices for idx in top_2k) / k

    return best_k, best_2k


def run_single_config(n_features, feat_drop, verbose=True):
    """Run a single hyperparameter configuration."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Generate dataset
    X, X_tilde, y = generate_dataset(DATASET, N_SAMPLES, n_features)

    # Apply preprocessing (scale to [-1, 1])
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    # 2-fold CV
    splits = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(X))
    best_ks = []
    best_2ks = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Shuffle features
        idx = np.arange(n_features)
        np.random.shuffle(idx)
        X_train, X_test = X_train[:, idx], X_test[:, idx]

        # True informative features after shuffling
        true_informative_indices = set(np.where(idx < K)[0].tolist())

        try:
            # Run ADMM with custom feat_drop
            _, _, scores, _ = run_admm_input_group(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                n_classes=2,
                seed=SEED + fold_idx,
                hp_overrides={"feat_drop": feat_drop, **HP_FIXED},
            )

            if scores is not None:
                best_k, best_2k = compute_best_k_scores(
                    scores, true_informative_indices, K
                )
                best_ks.append(best_k)
                best_2ks.append(best_2k)

        except Exception as e:
            if verbose:
                print(f"    Fold {fold_idx} failed: {e}")
            continue

    # Compute average metrics
    avg_best_k = np.mean(best_ks) if best_ks else 0.0
    avg_best_2k = np.mean(best_2ks) if best_2ks else 0.0

    return {
        "best_k": avg_best_k,
        "best_2k": avg_best_2k,
        "per_fold_bestk": best_ks,
        "per_fold_best2k": best_2ks,
        "success": len(best_ks) == CV_FOLDS,
    }


def main():
    """Run hyperparameter tuning experiment."""
    print("=" * 70)
    print("Hyperparameter Tuning: Feature Dropout Ablation")
    print("=" * 70)
    print("")
    print(f"Dataset: {DATASET} (k={K})")
    print(f"Samples: {N_SAMPLES}")
    print(f"CV Folds: {CV_FOLDS}")
    print(f"Feature dimensions: {FEATURE_DIMS}")
    print(f"Feature dropout values: {FEAT_DROP_VALUES}")
    print("")
    print("Fixed hyperparameters:")
    for key, val in HP_FIXED.items():
        print(f"  {key}: {val}")
    print("")

    all_results = {}

    for feat_drop in FEAT_DROP_VALUES:
        print(f"\n{'='*60}")
        print(f"Testing feat_drop = {feat_drop}")
        print(f"{'='*60}")

        feat_drop_results = {
            "feat_drop": feat_drop,
            "per_m": [],
            "avg_best_k": 0.0,
        }

        m_results = []
        for n_features in FEATURE_DIMS:
            t0 = time.time()
            result = run_single_config(n_features, feat_drop)
            elapsed = time.time() - t0

            m_results.append({
                "m": n_features,
                "best_k": result["best_k"],
                "best_2k": result["best_2k"],
                "per_fold_bestk": result["per_fold_bestk"],
                "time_seconds": elapsed,
            })

            status = "OK" if result["success"] else "PARTIAL"
            print(f"  m={n_features:5d}: best_k={result['best_k']:.4f}  "
                  f"({status}, {elapsed:.1f}s)")

        feat_drop_results["per_m"] = m_results
        feat_drop_results["avg_best_k"] = np.mean([r["best_k"] for r in m_results])

        all_results[feat_drop] = feat_drop_results
        print(f"  --> avg_best_k = {feat_drop_results['avg_best_k']:.4f}")

    # Find best configuration
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("")
    print(f"{'feat_drop':>12} | {'avg_best_k':>12} | {'Delta vs 0.7':>15}")
    print(f"{'-'*12}-+-{'-'*12}-+-{'-'*15}")

    # Reference: current performance with feat_drop=0.7 (from comparison)
    REFERENCE_RING_AVG_BEST_K = 0.12037037037037036  # From comparison_old_vs_new.json

    best_feat_drop = max(all_results.keys(), key=lambda x: all_results[x]["avg_best_k"])
    best_avg_best_k = all_results[best_feat_drop]["avg_best_k"]

    for feat_drop in sorted(all_results.keys()):
        avg_k = all_results[feat_drop]["avg_best_k"]
        delta = avg_k - REFERENCE_RING_AVG_BEST_K
        marker = " <-- BEST" if feat_drop == best_feat_drop else ""
        print(f"{feat_drop:>12.1f} | {avg_k:>12.4f} | {delta:>+15.4f}{marker}")

    print(f"\nBest configuration: feat_drop = {best_feat_drop}")
    print(f"Best avg_best_k: {best_avg_best_k:.4f}")
    print(f"Reference (feat_drop=0.7): {REFERENCE_RING_AVG_BEST_K:.4f}")
    print(f"Improvement: {best_avg_best_k - REFERENCE_RING_AVG_BEST_K:+.4f}")

    # Save results
    output = {
        "dataset": DATASET,
        "n_samples": N_SAMPLES,
        "cv_folds": CV_FOLDS,
        "fixed_hyperparameters": HP_FIXED,
        "tested_feat_drop_values": FEAT_DROP_VALUES,
        "tested_feature_dims": FEATURE_DIMS,
        "results": all_results,
        "best_configuration": {
            "feat_drop": best_feat_drop,
            "avg_best_k": best_avg_best_k,
        },
        "reference_performance": {
            "feat_drop": 0.7,
            "avg_best_k": REFERENCE_RING_AVG_BEST_K,
        },
    }

    output_file = os.path.join(results_dir, "feat_drop_ablation_ring.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == "__main__":
    main()
