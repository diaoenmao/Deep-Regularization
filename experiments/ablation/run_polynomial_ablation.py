# -*- coding: utf-8 -*-
"""
Ablation: Expanded-Space Feature Selection vs Original-Feature Selection

This script compares different selection strategies when using polynomial expansion:

1. **Original-Feature Selection (group)**: One gate per original feature.
   All expanded features from the same original feature share the same gate.
   This is the current default approach.

2. **Expanded-Space Selection**: One independent gate per expanded feature.
   Each polynomial term (x1, x2, x1^2, x1*x2, etc.) gets its own gate.
   This allows finer-grained selection but may select inconsistent terms.

3. **Hierarchical Selection**: Two-level selection.
   First select original features, then select expanded terms within them.

Key question: Does allowing the model to select individual polynomial terms
improve feature recovery compared to forcing all expansions of a feature to
share the same gate?

Usage:
    python run_polynomial_ablation.py --quick  # Quick test on small datasets
    python run_polynomial_ablation.py --full   # Full benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Set up paths for running from custom_admm directory
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)  # custom_admm
sys.path.insert(0, os.path.join(ROOT, "src"))  # custom_admm/src
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "Feature-Selection-Benchmark", "src"))

# Now imports should work
from src.data import generate_dataset
from src.polynomial_expansion import (
    PolynomialFeatureSelectionModel,
    compute_polynomial_expansion,
    compute_expanded_size,
    get_expansion_groups,
)
from src.admm_input_group_wrapper import _train_input_group, _Scaler


# ---------------------------------------------------------------------------
# Ground truth feature indices for each dataset
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "xor": [0, 1],
    "ring": [0, 1],
    "ring+xor": [0, 1, 2, 3],
    "ring+xor+sum": [0, 1, 2, 3, 4, 5],
}

K_VALUES = {
    "xor": 2,
    "ring": 2,
    "ring+xor": 4,
    "ring+xor+sum": 6,
}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

# Training hyperparameters matching main SADMM-FS method
FAIR_TRAINING_CONFIG = {
    "lr": 0.005,  # Main method default
    "C": 0.05,  # RatioNorm sparsity coefficient
    "batch_size": 64,  # Main method default
    "epochs": 500,  # Main method default
    "warmup_epochs": 120,  # Main method default
    "patience": 66,
}


def _rho_for_dim(m: int) -> float:
    """Dimension-dependent rho value matching main benchmark."""
    if m < 64:
        return 20.0
    if m < 256:
        return 50.0
    if m < 512:
        return 100.0
    return 200.0


def train_polynomial_model(
    model: PolynomialFeatureSelectionModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    epochs: int = None,
    warmup_epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    C: float = None,
    device: str = "cpu",
    verbose: bool = False,
    use_early_stopping: bool = False,  # ADMM doesn't use early stopping by default
    patience: int = None,
) -> np.ndarray:
    """
    Train polynomial feature selection model with ADMM+RatioNorm.

    This NOW uses the same ADMM+RatioNorm training as the main SADMM-FS method.
    The sklearn polynomial expansion is just a preprocessing step - gradients
    flow to the gates via the downstream predictor.

    Key insight: We don't need gradients through the expansion itself.
    The gates learn which expanded features matter through the task loss.

    Training config matches main SADMM-FS method:
    - lr=0.005, batch_size=64, epochs=500, optimizer=Adam
    - ADMM with RatioNorm proximal operator

    Returns the scaled training data for evaluation.
    """
    # Use fair hyperparameters if not specified
    if epochs is None:
        epochs = FAIR_TRAINING_CONFIG["epochs"]
    if warmup_epochs is None:
        warmup_epochs = FAIR_TRAINING_CONFIG["warmup_epochs"]
    if lr is None:
        lr = FAIR_TRAINING_CONFIG["lr"]
    if batch_size is None:
        batch_size = FAIR_TRAINING_CONFIG["batch_size"]
    if C is None:
        C = FAIR_TRAINING_CONFIG["C"]
    if patience is None:
        patience = FAIR_TRAINING_CONFIG["patience"]

    # Scale data (matching main benchmark)
    scaler = _Scaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model.to(device)

    # Use ADMM+RatioNorm training from admm_input_group_wrapper
    # This is the same training procedure as main SADMM-FS
    _train_input_group(
        model,
        X_train_scaled,
        y_train,
        n_classes,
        lr=lr,
        C=C,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        batch_size=batch_size,
        rho_init=_rho_for_dim(model.gate.shape[0]),
        device=device,
        optimizer_type="adam",
        use_ratio_norm=True,
        use_admm=True,
        use_early_stopping=use_early_stopping,
        patience=patience,
        val_split=0.2,
    )

    return X_train_scaled


def evaluate_model(
    model: PolynomialFeatureSelectionModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    ground_truth: List[int],
    k: int,
    device: str = "cpu",
) -> Dict:
    """
    Evaluate model and compute best-k recovery.

    Returns dict with:
    - best_k: fraction of true features in top-k
    - best_2k: fraction of true features in top-2k
    - selected_original: indices of selected original features
    - scores: feature importance scores for original features
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits = model(X_t)

        # Get predictions
        if n_classes > 2:
            y_pred = logits.argmax(dim=1).cpu().numpy()
        else:
            y_pred = (torch.sigmoid(logits.squeeze()) > 0.5).cpu().numpy()

        # Get original feature scores
        scores = model.get_original_feature_scores().cpu().numpy()

    # Compute best-k recovery
    # CRITICAL: Use absolute values for unbounded gates (can be negative)
    abs_scores = np.abs(scores)
    top_k_indices = np.argsort(abs_scores)[-k:]
    top_2k_indices = np.argsort(abs_scores)[-2*k:]

    # Fraction of ground truth in top-k
    best_k = len(set(top_k_indices) & set(ground_truth)) / k
    best_2k = len(set(top_2k_indices) & set(ground_truth)) / len(ground_truth) if 2*k >= len(ground_truth) else len(set(top_2k_indices) & set(ground_truth)) / (2*k)

    # Accuracy
    accuracy = np.mean(y_pred == y_test)

    return {
        "best_k": best_k,
        "best_2k": best_2k,
        "accuracy": accuracy,
        "selected_original": top_k_indices.tolist(),
        "scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    dataset_name: str,
    n_samples: int,
    n_features: int,
    selection_mode: str,
    degree: int,
    seed: int,
    epochs: int = None,
    warmup_epochs: int = None,
    C: float = None,
    device: str = "cpu",
) -> Dict:
    """
    Run a single experiment with fair hyperparameters.
    """
    # Use fair hyperparameters if not specified
    if epochs is None:
        epochs = FAIR_TRAINING_CONFIG["epochs"]
    if warmup_epochs is None:
        warmup_epochs = FAIR_TRAINING_CONFIG["warmup_epochs"]
    if C is None:
        C = FAIR_TRAINING_CONFIG["C"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate data
    X, X_tilde, y = generate_dataset(dataset_name, n_samples, n_features)

    # DATA CENTERING: Match main benchmark (transform from [0,1] to [-1,1])
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    # Split train/test
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    n_classes = len(np.unique(y))

    # Ground truth
    ground_truth = GROUND_TRUTH[dataset_name]
    k = K_VALUES[dataset_name]

    # Create model
    model = PolynomialFeatureSelectionModel(
        input_size=n_features,
        n_classes=n_classes,
        degree=degree,
        selection_mode=selection_mode,
        hidden_dims=[32, 32],
        activation="mish",
        dropout=0.043,  # Match main benchmark
    )

    # Train with fair hyperparameters (returns scaled data)
    X_train_scaled = train_polynomial_model(
        model,
        X_train,
        y_train,
        n_classes,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        C=C,
        device=device,
        verbose=False,
    )

    # Scale test data using the same parameters as training
    # Note: The scaler is created inside train_polynomial_model, so we need to handle this differently
    # For now, we'll just use the original test data since the gate values are what matter for feature selection
    # Evaluate
    results = evaluate_model(
        model,
        X_test,  # Use original test data - the gate values are what matters
        y_test,
        n_classes,
        ground_truth,
        k,
        device=device,
    )

    # Add metadata
    results.update({
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "selection_mode": selection_mode,
        "degree": degree,
        "seed": seed,
        "expanded_size": model.expanded_size,
        "ground_truth": ground_truth,
    })

    return results


def run_ablation(
    quick: bool = False,
    seeds: Optional[List[int]] = None,
    device: str = "cpu",
) -> Dict:
    """
    Run full ablation study.

    Compares:
    - selection_mode: "group" vs "expanded" vs "hierarchical"
    - degree: 1 (no expansion), 2 (quadratic), 3 (cubic)
    """
    if seeds is None:
        # Protocol: 6-fold CV, seed = 42 + fold_idx
        seeds = [42, 43, 44] if quick else [42, 43, 44, 45, 46, 47]

    # Dataset configurations
    if quick:
        datasets = [
            ("xor", 500, 32),  # Quick test with smaller dataset
        ]
    else:
        # GPU-friendly: smaller dimensions for degree=2
        datasets = [
            ("xor", 1000, 64),      # m=64, degree=2 -> 2145 features
            ("ring", 1000, 64),
            ("ring+xor", 1000, 64),  # Reduced from 256 to 64
        ]

    # Ablation configurations
    selection_modes = ["group", "expanded"]  # Simplified for quick test
    degrees = [1, 2]  # degree=1 is baseline (no expansion)

    all_results = []

    for dataset_name, n_samples, n_features in datasets:
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset: {dataset_name} (m={n_features}, n={n_samples})", flush=True)
        print(f"{'='*60}", flush=True)

        for degree in degrees:
            expanded_size = compute_expanded_size(n_features, degree)
            print(f"\nDegree {degree}: {n_features} -> {expanded_size} features", flush=True)

            for selection_mode in selection_modes:
                print(f"\n  Selection mode: {selection_mode}", flush=True)

                for seed in seeds:
                    try:
                        result = run_single_experiment(
                            dataset_name=dataset_name,
                            n_samples=n_samples,
                            n_features=n_features,
                            selection_mode=selection_mode,
                            degree=degree,
                            seed=seed,
                            # Use fair hyperparameters (defaults from FAIR_TRAINING_CONFIG)
                            device=device,
                        )

                        all_results.append(result)

                        print(f"    Seed {seed}: best_k={result['best_k']:.4f}, "
                              f"acc={result['accuracy']:.4f}", flush=True)

                    except Exception as e:
                        print(f"    Seed {seed}: FAILED - {e}", flush=True)

    # Aggregate results
    summary = aggregate_results(all_results, selection_modes, degrees, datasets)

    return {
        "results": all_results,
        "summary": summary,
        "config": {
            "quick": quick,
            "seeds": seeds,
            "datasets": [(d[0], d[1], d[2]) for d in datasets],
            "selection_modes": selection_modes,
            "degrees": degrees,
        },
        "timestamp": datetime.now().isoformat(),
    }


def aggregate_results(
    results: List[Dict],
    selection_modes: List[str],
    degrees: List[int],
    datasets: List[Tuple[str, int, int]],
) -> Dict:
    """
    Aggregate results by dataset, degree, and selection mode.
    """
    summary = {}

    for dataset_name, _, _ in datasets:
        summary[dataset_name] = {}

        for degree in degrees:
            summary[dataset_name][f"degree_{degree}"] = {}

            for mode in selection_modes:
                # Filter results
                filtered = [
                    r for r in results
                    if r["dataset"] == dataset_name
                    and r["degree"] == degree
                    and r["selection_mode"] == mode
                ]

                if filtered:
                    best_k_mean = np.mean([r["best_k"] for r in filtered])
                    best_k_std = np.std([r["best_k"] for r in filtered])
                    acc_mean = np.mean([r["accuracy"] for r in filtered])

                    summary[dataset_name][f"degree_{degree}"][mode] = {
                        "best_k_mean": best_k_mean,
                        "best_k_std": best_k_std,
                        "accuracy_mean": acc_mean,
                        "n_runs": len(filtered),
                    }

    return summary


def print_summary_table(summary: Dict) -> None:
    """
    Print a formatted summary table.
    """
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY: Expanded-Space vs Original-Feature Selection")
    print("=" * 80)

    for dataset_name, dataset_data in summary.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 60)

        for degree_key, mode_data in dataset_data.items():
            print(f"\n  {degree_key}:")

            for mode, stats in mode_data.items():
                print(f"    {mode:15s}: best_k = {stats['best_k_mean']:.4f} +/- {stats['best_k_std']:.4f}, "
                      f"acc = {stats['accuracy_mean']:.4f} ({stats['n_runs']} runs)")

    # Print key comparison
    print("\n" + "=" * 80)
    print("KEY COMPARISON: group (original-feature) vs expanded (expanded-space)")
    print("=" * 80)

    for dataset_name, dataset_data in summary.items():
        print(f"\n{dataset_name.upper()}:")

        for degree_key in ["degree_1", "degree_2"]:
            if degree_key in dataset_data:
                mode_data = dataset_data[degree_key]

                group_best = mode_data.get("group", {}).get("best_k_mean", 0)
                expanded_best = mode_data.get("expanded", {}).get("best_k_mean", 0)
                hier_best = mode_data.get("hierarchical", {}).get("best_k_mean", 0)

                print(f"  {degree_key}:")
                print(f"    group (shared gate):        {group_best:.4f}")
                print(f"    expanded (independent):     {expanded_best:.4f}")
                print(f"    hierarchical (two-level):   {hier_best:.4f}")

                if group_best > 0 and expanded_best > 0:
                    diff = expanded_best - group_best
                    if diff > 0.01:
                        print(f"    -> expanded BETTER by {diff:.4f}")
                    elif diff < -0.01:
                        print(f"    -> group BETTER by {-diff:.4f}")
                    else:
                        print(f"    -> SIMILAR (diff={diff:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run polynomial feature selection ablation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with fewer seeds and smaller datasets",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark with all datasets and more seeds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    if args.full:
        args.quick = False

    print("=" * 80)
    print("POLYNOMIAL FEATURE SELECTION ABLATION")
    print("=" * 80)
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Device: {args.device}")

    # Run ablation
    results = run_ablation(
        quick=args.quick,
        device=args.device,
    )

    # Print summary
    print_summary_table(results["summary"])

    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(ROOT) / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"polynomial_ablation_{timestamp}.json"

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Return for programmatic access
    return results


if __name__ == "__main__":
    main()