#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adaptive Architecture Comparison Experiment.

This script compares different model architectures for feature selection:
1. Small (2-layer, 32 units) - LassoNet style
2. Medium (3-layer, 48 units) - Compromise
3. Baseline (5-layer, 58 units) - Current standard
4. Adaptive (dimension-aware gate) - Proposed

Key Hypothesis:
- Adaptive model performs well on both low-dim and high-dim tasks
- Medium model provides intermediate performance

Usage:
    python run_adaptive_comparison.py --all          # Run all experiments
    python run_adaptive_comparison.py --quick        # Quick test
    python run_adaptive_comparison.py --xor          # XOR only
    python run_adaptive_comparison.py --ring         # Ring only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

# Add paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from src.adaptive_architecture import (
    AdaptiveFeatureSelectionMLP,
    AdaptiveFeatureSelector,
    get_adaptive_capacity,
    get_adaptive_depth,
)
from src.admm_input_group_wrapper import (
    GatedFeatureSelectionMLP,
    _extract_feature_importance,
    _Scaler,
    _train_adaptive_input_group,
    _train_input_group,
)
from src.data import generate_dataset
from src.nn_wrapper import MODEL_PRESETS, Model, create_model_with_preset

# Constants
SEED = 0
N_SAMPLES = 1000
N_FOLDS = 6
RESULTS_DIR = os.path.join(ROOT, "results", "adaptive_comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Architecture configurations
ARCHITECTURES = {
    "small_2layer": {
        "desc": "Small: 2-layer × 32 units (LassoNet style)",
        "type": "preset",
        "preset": "small_2layer",
    },
    "medium_3layer": {
        "desc": "Medium: 3-layer × 48 units (compromise)",
        "type": "preset",
        "preset": "medium_3layer",
    },
    "baseline_5layer": {
        "desc": "Baseline: 5-layer × 58 units (current standard)",
        "type": "preset",
        "preset": "baseline_5layer",
    },
    "adaptive_mlp": {
        "desc": "Adaptive MLP: capacity scales with √(m·k)",
        "type": "adaptive_mlp",
    },
    "adaptive_gate": {
        "desc": "Adaptive Gate: √m bottleneck + fixed backbone",
        "type": "adaptive_gate",
    },
}


def create_model(
    arch_name: str, n_features: int, n_classes: int = 2
) -> torch.nn.Module:
    """Create model based on architecture name."""
    arch = ARCHITECTURES[arch_name]

    if arch["type"] == "preset":
        # Use GatedFeatureSelectionMLP for preset architectures
        preset = MODEL_PRESETS[arch["preset"]]
        return GatedFeatureSelectionMLP(
            input_size=n_features,
            n_classes=n_classes,
            latent_size=preset["latent_size"],
            n_hidden_layers=preset["n_hidden_layers"],
            gaussian_noise=preset["gaussian_noise"],
            dropout=preset["dropout"],
            activation=preset["activation"],
            feat_drop=0.6,
        )

    elif arch["type"] == "adaptive_mlp":
        # Adaptive capacity based on Capacity-Dimension Matching
        latent_size = get_adaptive_capacity(n_features, k_estimate=4)
        n_hidden_layers = get_adaptive_depth(n_features)
        return AdaptiveFeatureSelectionMLP(
            input_size=n_features,
            n_classes=n_classes,
            latent_size=latent_size,
            n_hidden_layers=n_hidden_layers,
            feat_drop=0.6,
            dropout=0.043,
        )

    elif arch["type"] == "adaptive_gate":
        return AdaptiveFeatureSelector(
            n_features=n_features,
            n_classes=n_classes,
            backbone_dims=[64, 32],
            feat_drop=0.6,
            dropout=0.0,
        )

    else:
        raise ValueError(f"Unknown architecture type: {arch['type']}")


def evaluate_bestk(scores: np.ndarray, k_true: int) -> float:
    """Compute best-k feature selection accuracy."""
    top_idx = set(np.argsort(np.abs(scores))[-k_true:])
    true_set = set(range(k_true))
    return len(top_idx & true_set) / k_true


def run_single_fold(
    arch_name: str,
    ds_name: str,
    k_true: int,
    n_features: int,
    fold_idx: int,
    device: str = "cpu",
    quick: bool = False,
) -> Dict:
    """Run single fold of cross-validation."""
    # Set seeds
    np.random.seed(SEED + fold_idx)
    torch.manual_seed(SEED + fold_idx)

    # Generate data
    X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, n_features)
    X = 2.0 * X - 1.0  # Scale to [-1, 1]

    # Split for CV
    splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X))
    train_idx, test_idx = splits[fold_idx]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Shuffle features (as in main benchmark)
    idx = np.arange(n_features)
    np.random.shuffle(idx)
    X_train, X_test = X_train[:, idx], X_test[:, idx]
    correct = set(np.where(idx < k_true)[0].tolist())

    # Standardize
    scaler = _Scaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Create model
    model = create_model(arch_name, n_features, n_classes=2)
    model = model.to(device)

    # Training parameters
    epochs = 150 if quick else 500
    warmup_epochs = 50 if quick else 120

    # Train
    try:
        if arch_name in ["adaptive_mlp", "adaptive_gate"]:
            _train_adaptive_input_group(
                model,
                X_train_s,
                y_train,
                n_classes=2,
                lr=0.005,
                C=0.05,
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                device=device,
                use_early_stopping=not quick,
            )
        else:
            _train_input_group(
                model,
                X_train_s,
                y_train,
                n_classes=2,
                lr=0.005,
                C=0.05,
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                device=device,
                use_early_stopping=not quick,
                n_features=n_features,
            )
    except Exception as e:
        print(f"  Training failed for {arch_name}: {e}")
        return None

    # Extract feature importance
    scores = _extract_feature_importance(model, X_train_s)

    # Unshuffle scores to match original feature order
    scores_unshuffled = np.zeros_like(scores)
    scores_unshuffled[idx] = scores

    # Compute best-k
    best_k = evaluate_bestk(scores_unshuffled, k_true)

    # Compute test AUC
    model.eval()
    with torch.no_grad():
        x_test = torch.FloatTensor(X_test_s).to(device)
        logits = model(x_test).cpu().numpy().flatten()
        try:
            auc = roc_auc_score(y_test, logits)
        except:
            auc = 0.5

    return {
        "best_k": best_k,
        "auc": auc,
        "scores": scores_unshuffled.tolist(),
    }


def run_architecture_comparison(
    arch_name: str,
    ds_name: str,
    k_true: int,
    dimensions: List[int],
    device: str = "cpu",
    quick: bool = False,
) -> Dict:
    """Run full comparison for one architecture on one dataset."""
    print(f"\n  Running {ARCHITECTURES[arch_name]['desc']} on {ds_name}")

    results = {"dimensions": {}}

    for n_features in dimensions:
        print(f"    m={n_features:5d}...", end=" ", flush=True)

        fold_results = []
        for fold_idx in range(N_FOLDS):
            result = run_single_fold(
                arch_name, ds_name, k_true, n_features, fold_idx, device, quick
            )
            if result:
                fold_results.append(result)

        if fold_results:
            best_ks = [r["best_k"] for r in fold_results]
            aucs = [r["auc"] for r in fold_results]

            mean_best_k = np.mean(best_ks)
            std_best_k = np.std(best_ks)
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            results["dimensions"][str(n_features)] = {
                "mean_best_k": mean_best_k,
                "std_best_k": std_best_k,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "per_fold_best_k": best_ks,
                "per_fold_auc": aucs,
            }

            print(
                f"best-k={mean_best_k:.1%}±{std_best_k:.1%}, AUC={mean_auc:.3f}±{std_auc:.3f}"
            )
        else:
            print("FAILED")
            results["dimensions"][str(n_features)] = {
                "mean_best_k": 0.0,
                "std_best_k": 0.0,
                "mean_auc": 0.5,
                "std_auc": 0.0,
            }

    return results


def run_all_experiments(
    quick: bool = False,
    device: str = "cpu",
    target_datasets: Optional[List[str]] = None,
) -> Dict:
    """Run full architecture comparison."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if quick else "full"

    results = {
        "metadata": {
            "timestamp": timestamp,
            "mode": mode,
            "device": device,
            "n_samples": N_SAMPLES,
            "n_folds": N_FOLDS,
        },
        "architectures": {},
    }

    # Dataset configurations
    datasets = {
        "xor": (2, [8, 128, 1024] if not quick else [8, 128]),
        "ring": (2, [32, 512] if not quick else [32]),
        "ring+xor": (4, [16, 256] if not quick else [16]),
    }

    if target_datasets:
        datasets = {k: v for k, v in datasets.items() if k in target_datasets}

    # Run for each architecture
    for arch_name in ARCHITECTURES.keys():
        print(f"\n{'=' * 60}")
        print(f"  Architecture: {arch_name}")
        print(f"  {ARCHITECTURES[arch_name]['desc']}")
        print(f"{'=' * 60}")

        arch_results = {"datasets": {}}

        for ds_name, (k_true, dimensions) in datasets.items():
            ds_result = run_architecture_comparison(
                arch_name, ds_name, k_true, dimensions, device, quick
            )
            arch_results["datasets"][ds_name] = ds_result

        results["architectures"][arch_name] = arch_results

        # Save intermediate results
        save_path = os.path.join(
            RESULTS_DIR, f"adaptive_comparison_{timestamp}_{mode}.json"
        )
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [Saved] {save_path}")

    return results


def print_summary(results: Dict) -> None:
    """Print summary table."""
    print("\n" + "=" * 80)
    print("  SUMMARY: Architecture Comparison")
    print("=" * 80)

    architectures = list(results["architectures"].keys())
    datasets = list(list(results["architectures"].values())[0]["datasets"].keys())

    for ds_name in datasets:
        print(f"\n  Dataset: {ds_name}")
        print("-" * 80)
        print(f"  {'Arch':<20} {'m':<8} {'best-k':<12} {'AUC':<12}")
        print("-" * 80)

        for arch_name in architectures:
            arch_data = results["architectures"][arch_name]["datasets"][ds_name]
            for dim_str, dim_data in arch_data["dimensions"].items():
                best_k = dim_data["mean_best_k"]
                auc = dim_data["mean_auc"]
                marker = ""
                if best_k >= 0.9:
                    marker = "*"
                elif best_k < 0.3:
                    marker = "x"
                print(
                    f"  {arch_name:<20} {dim_str:<8} {best_k:.1%} {marker:<4} {auc:.3f}"
                )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Architecture Comparison")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--xor", action="store_true", help="XOR dataset only")
    parser.add_argument("--ring", action="store_true", help="Ring dataset only")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    # Determine target datasets
    target_datasets = None
    if args.xor:
        target_datasets = ["xor"]
    elif args.ring:
        target_datasets = ["ring"]

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Running adaptive architecture comparison")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print(f"  Device: {device}")
    print(f"  Target datasets: {target_datasets or 'all'}")

    # Run experiments
    results = run_all_experiments(
        quick=args.quick, device=device, target_datasets=target_datasets
    )

    # Print summary
    print_summary(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
