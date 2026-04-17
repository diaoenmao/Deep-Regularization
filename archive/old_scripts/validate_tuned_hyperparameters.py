#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate updated hyperparameters on full benchmark.

This script runs the full benchmark with the TUNED hyperparameters
(feat_drop=0.6) and compares with the previous results.

Expected improvement based on Phase 1 tuning:
- ring: avg_best_k from 0.12 -> ~0.30 (based on m=[32,64,128,256] subset)
- Other datasets: variable improvement expected
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

SEED = 0
N_SAMPLES = 1000
METHOD = "admm_input_group_tuned"

# Dataset configuration - QUICK VALIDATION (fewer feature dims for speed)
datasets_config = [
    ("xor",          2, [32, 64, 128, 256]),      # Subset for quick validation
    ("ring",         2, [32, 64, 128, 256]),      # Key dataset for tuning validation
    ("ring+xor",     4, [32, 64, 128]),           # Subset
    ("ring+xor+sum", 6, [32, 64, 128]),           # Subset
]

CV_FOLDS = 2  # Use 2-fold for quick validation (not 6-fold)

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "validation")
os.makedirs(results_dir, exist_ok=True)


def compute_best_k_scores(scores, true_informative_indices, k):
    """Compute best_k and best_2k scores."""
    ranked = np.argsort(np.abs(scores))[::-1]
    top_k = ranked[:k]
    best_k = sum(idx in true_informative_indices for idx in top_k) / k
    top_2k = ranked[:2*k]
    best_2k = sum(idx in true_informative_indices for idx in top_2k) / k
    return best_k, best_2k


def main():
    """Run validation experiment."""
    print("=" * 70)
    print("Hyperparameter Validation - Updated Defaults (feat_drop=0.6)")
    print("=" * 70)
    print("")
    print(f"Method: {METHOD}")
    print(f"Samples: {N_SAMPLES}")
    print(f"CV Folds: {CV_FOLDS}")
    print("")
    print("Updated hyperparameters:")
    print("  feat_drop: 0.6 (tuned from 0.7)")
    print("  lr: 0.005")
    print("  C: 0.05")
    print("  epochs: 500")
    print("  warmup_epochs: 120")
    print("  rho_init: 200.0")
    print("")

    # Load previous results for comparison
    prev_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "results", "comparison_old_vs_new.json")
    try:
        with open(prev_results_path, "r") as f:
            prev_comparison = json.load(f)
        print(f"Previous results loaded: {prev_results_path}")
    except FileNotFoundError:
        prev_comparison = None
        print("No previous results found for comparison")
    print("")

    all_results = {}
    total_start = time.time()

    for ds_name, k, feature_dims in datasets_config:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}  (k={k})")
        print(f"{'='*60}")

        dataset_results = {
            "avg_best_k": 0.0,
            "details": []
        }

        for n_features in feature_dims:
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, n_features)
            X = 2.0 * X - 1.0
            X_tilde = 2.0 * X_tilde - 1.0

            splits = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(X))
            best_ks = []
            best_2ks = []
            fold_times = []

            t0 = time.time()
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                idx = np.arange(n_features)
                np.random.shuffle(idx)
                X_train, X_test = X_train[:, idx], X_test[:, idx]
                true_informative_indices = set(np.where(idx < k)[0].tolist())

                try:
                    _, _, scores, _ = run_admm_input_group(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        n_classes=2,
                        seed=SEED + fold_idx,
                    )

                    if scores is not None:
                        best_k, best_2k = compute_best_k_scores(
                            scores, true_informative_indices, k
                        )
                        best_ks.append(best_k)
                        best_2ks.append(best_2k)

                except Exception as e:
                    print(f"    Fold {fold_idx} failed: {e}")
                    continue

            elapsed = time.time() - t0

            bk = np.mean(best_ks) if best_ks else 0
            b2k = np.mean(best_2ks) if best_2ks else 0

            print(f"  m={n_features:5d}: best_k={bk:.4f}  ({elapsed:.1f}s)")

            dataset_results["details"].append({
                "m": n_features,
                "best_k": float(bk),
                "best_2k": float(b2k),
                "time_seconds": elapsed,
            })

        # Compute overall avg_best_k for this dataset
        if dataset_results["details"]:
            dataset_results["avg_best_k"] = np.mean(
                [d["best_k"] for d in dataset_results["details"]]
            )

        all_results[ds_name] = dataset_results
        print(f"  --> Dataset avg_best_k: {dataset_results['avg_best_k']:.4f}")

    total_elapsed = time.time() - total_start

    # Summary comparison
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print("")
    print(f"Total validation time: {total_elapsed:.1f}s")
    print("")
    print(f"{'Dataset':>15} | {'Tuned avg_best_k':>18} | {'Feature dims tested':>20}")
    print(f"{'-'*15}-+-{'-'*18}-+-{'-'*20}")

    for ds_name, results in all_results.items():
        dims = [d["m"] for d in results["details"]]
        print(f"{ds_name:>15} | {results['avg_best_k']:>18.4f} | {str(dims):>20}")

    # Compare with previous if available
    if prev_comparison:
        print(f"\n{'='*70}")
        print("COMPARISON WITH PREVIOUS (feat_drop=0.7)")
        print(f"{'='*70}")
        print("")
        print(f"{'Dataset':>15} | {'Previous':>10} | {'Tuned':>10} | {'Delta':>10}")
        print(f"{'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for ds_name, results in all_results.items():
            tuned_k = results["avg_best_k"]
            # Get previous from comparison file (this is approx - uses different feature dims)
            if ds_name in prev_comparison:
                prev_details = prev_comparison[ds_name].get("details", [])
                # Average over overlapping m values
                prev_m_values = {d["m"]: d["best_k"] for d in prev_details}
                overlapping_m = [m for m in [32, 64, 128, 256] if m in prev_m_values]
                if overlapping_m:
                    prev_k = np.mean([prev_m_values[m] for m in overlapping_m])
                else:
                    prev_k = prev_comparison[ds_name].get("avg_best_k", 0)
            else:
                prev_k = 0

            delta = tuned_k - prev_k
            print(f"{ds_name:>15} | {prev_k:>10.4f} | {tuned_k:>10.4f} | {delta:>+10.4f}")

    # Save validation results
    output = {
        "method": METHOD,
        "n_samples": N_SAMPLES,
        "cv_folds": CV_FOLDS,
        "hyperparameters": {
            "feat_drop": 0.6,
            "lr": 0.005,
            "C": 0.05,
            "epochs": 500,
            "warmup_epochs": 120,
            "rho_init": 200.0,
        },
        "datasets": all_results,
        "total_time_seconds": total_elapsed,
    }

    output_file = os.path.join(results_dir, "validation_tuned_hyperparameters.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nValidation results saved to: {output_file}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    main()
