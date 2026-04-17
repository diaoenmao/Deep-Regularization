#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal method-focused ablation for SADMM-FS.

This isolates two design choices before rerunning broader benchmarks:
1. column-normalized first layer on/off
2. feature dropout on/off
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.admm_input_group_wrapper import run_admm_input_group
from src.data import generate_dataset

SEED = 0
N_SAMPLES = 1000
N_FOLDS = 6
RESULTS_DIR = os.path.join(ROOT, "results", "method_ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)

QUICK_CONFIG = {
    "xor": (2, [128]),
    "ring": (2, [32]),
    "ring+xor": (4, [16]),
}

FULL_CONFIG = {
    "xor": (2, [32, 128, 512]),
    "ring": (2, [32, 128]),
    "ring+xor": (4, [16, 64]),
}

VARIANTS = {
    "baseline_drop": {
        "label": "Baseline gate + feature dropout",
        "hp_overrides": {
            "feat_drop": 0.6,
            "column_normalize_first_layer": False,
        },
    },
    "baseline_no_drop": {
        "label": "Baseline gate without feature dropout",
        "hp_overrides": {
            "feat_drop": 0.0,
            "column_normalize_first_layer": False,
        },
    },
    "norm_drop": {
        "label": "Column-normalized gate + feature dropout",
        "hp_overrides": {
            "feat_drop": 0.6,
            "column_normalize_first_layer": True,
        },
    },
    "norm_no_drop": {
        "label": "Column-normalized gate without feature dropout",
        "hp_overrides": {
            "feat_drop": 0.0,
            "column_normalize_first_layer": True,
        },
    },
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def compute_best_k(scores: np.ndarray, correct: set[int], k: int) -> float:
    ranking = np.argsort(np.abs(scores))[::-1]
    return float(sum(idx in correct for idx in ranking[:k]) / k)


def prepare_fold(dataset_name: str, k: int, n_features: int, fold_idx: int) -> dict:
    set_seed(SEED + n_features)
    X, _, y = generate_dataset(dataset_name, N_SAMPLES, n_features)
    X = 2.0 * X - 1.0

    splits = list(KFold(n_splits=N_FOLDS).split(X))
    train_idx, test_idx = splits[fold_idx]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]

    rng = np.random.RandomState(SEED + n_features * 10 + fold_idx)
    feature_order = np.arange(n_features)
    rng.shuffle(feature_order)

    X_train = X_train[:, feature_order]
    X_test = X_test[:, feature_order]
    correct = set(np.where(feature_order < k)[0].tolist())

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "correct": correct,
        "k": k,
        "n_features": n_features,
    }


def run_variant_on_fold(fold: dict, hp_overrides: dict) -> dict:
    t0 = time.time()
    _, _, scores, _ = run_admm_input_group(
        fold["X_train"],
        fold["y_train"],
        fold["X_test"],
        n_classes=2,
        hp_overrides=hp_overrides,
        seed=SEED,
    )
    return {
        "best_k": compute_best_k(scores, fold["correct"], fold["k"]),
        "runtime_sec": time.time() - t0,
    }


def summarise(results: dict) -> dict:
    summary = {}
    for variant_name, datasets in results.items():
        all_scores = []
        dataset_summary = {}
        for dataset_name, dims in datasets.items():
            dim_scores = []
            dim_runtimes = []
            for payload in dims.values():
                fold_scores = [fold["best_k"] for fold in payload["folds"]]
                fold_runtimes = [fold["runtime_sec"] for fold in payload["folds"]]
                payload["mean_best_k"] = float(np.mean(fold_scores))
                payload["std_best_k"] = float(np.std(fold_scores))
                payload["mean_runtime_sec"] = float(np.mean(fold_runtimes))
                dim_scores.extend(fold_scores)
                dim_runtimes.extend(fold_runtimes)
            dataset_summary[dataset_name] = {
                "mean_best_k": float(np.mean(dim_scores)),
                "mean_runtime_sec": float(np.mean(dim_runtimes)),
            }
            all_scores.extend(dim_scores)
        summary[variant_name] = {
            "overall_mean_best_k": float(np.mean(all_scores)),
            "datasets": dataset_summary,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Method simplification ablation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fold-limit", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    args = parser.parse_args()

    config = QUICK_CONFIG if args.quick else FULL_CONFIG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "metadata": {
            "timestamp": timestamp,
            "quick_mode": args.quick,
            "fold_limit": args.fold_limit,
            "variants": {name: VARIANTS[name] for name in args.variants},
        },
        "results": {},
    }

    for variant_name in args.variants:
        variant_hp = VARIANTS[variant_name]["hp_overrides"]
        results["results"][variant_name] = {}
        for dataset_name, (k, dims) in config.items():
            results["results"][variant_name][dataset_name] = {}
            for n_features in dims:
                folds = []
                for fold_idx in range(min(args.fold_limit, N_FOLDS)):
                    fold = prepare_fold(dataset_name, k, n_features, fold_idx)
                    fold_result = run_variant_on_fold(fold, variant_hp)
                    fold_result["fold_idx"] = fold_idx
                    folds.append(fold_result)
                results["results"][variant_name][dataset_name][str(n_features)] = {
                    "k": k,
                    "folds": folds,
                }

    results["summary"] = summarise(results["results"])
    out_path = os.path.join(RESULTS_DIR, f"method_simplification_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
