# -*- coding: utf-8 -*-
"""
Run Full Benchmark with Best Gradual ADMM + Polynomial Expansion.

Uses the best configuration + polynomial degree=2 to improve Ring prediction.
"""

import os
import sys
import json
import importlib
from datetime import datetime
from pathlib import Path

_script_path = Path(__file__).resolve() if '__file__' in dir() else Path.cwd()
PROJECT_ROOT = _script_path.parent.parent.parent.resolve()
sys.path = [str(PROJECT_ROOT), str(PROJECT_ROOT / "Feature-Selection-Benchmark")] + sys.path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

src_gradual = importlib.import_module('src.gradual_admm_with_pruning')
gradual_admm_with_pruning = src_gradual.gradual_admm_with_pruning

src_admm = importlib.import_module('src.admm_input_group_wrapper')
GatedFeatureSelectionMLP = src_admm.GatedFeatureSelectionMLP
_Scaler = src_admm._Scaler

src_data = importlib.import_module('src.data')
generate_dataset = src_data.generate_dataset


def create_synthetic_dataset_with_poly(name: str, n_samples: int = 1000, n_features: int = 128, degree: int = 2, seed: int = 42):
    """Create synthetic dataset with polynomial feature expansion.

    Key insight:
    - XOR boundary is LINEAR (diagonal lines), so no squared terms needed
    - Ring boundary is CIRCULAR (x1² + x2² = r²), so squared terms needed
    """
    np.random.seed(seed)

    X, X_tilde, y = generate_dataset(name, n_samples=n_samples, n_features=n_features)

    # Scale data BEFORE polynomial expansion
    scaler = _Scaler()
    X_scaled = scaler.fit_transform(X)

    # Apply polynomial expansion
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    n_poly_features = X_poly.shape[1]

    # Build expanded ground truth based on dataset type
    # XOR: linear boundary -> no squared terms
    # Ring: circular boundary -> need squared terms
    if name == "xor":
        # XOR: features 0, 1 only (no squared terms needed)
        expanded_ground_truth = [0, 1]
        expanded_k = 2
    elif name == "ring":
        # Ring: features 0, 1 + their squares (128, 129)
        expanded_ground_truth = [0, 1, 128, 129]
        expanded_k = 4
    elif name == "ring+xor":
        # Ring+XOR: Ring(0,1) needs squares, XOR(2,3) doesn't
        # Ring part: [0, 1, 128, 129]
        # XOR part: [2, 3] (no squares)
        expanded_ground_truth = [0, 1, 2, 3, 128, 129]
        expanded_k = 6
    elif name == "ring+xor+sum":
        # Same as ring+xor: Ring needs squares, XOR/Sum don't
        expanded_ground_truth = [0, 1, 2, 3, 128, 129]
        expanded_k = 6
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"Polynomial expansion: {n_features} -> {n_poly_features} features")
    print(f"Expanded ground truth: {expanded_ground_truth} (k={expanded_k})")

    return {
        "name": name,
        "X": X_poly,
        "y": y,
        "n_classes": 2,
        "ground_truth": expanded_ground_truth,
        "k": expanded_k,
        "n_features": n_poly_features,
        "n_samples": n_samples,
        "original_n_features": n_features,
        "degree": degree,
    }


def compute_best_k(selected: list, ground_truth: list, k: int) -> float:
    """Compute best-k metric."""
    return len(set(selected) & set(ground_truth)) / k


def group_selection_from_gate(gate: torch.Tensor, n_original: int, k_original: int, degree: int = 2) -> list:
    """
    Group selection for polynomial features.

    Instead of selecting individual features, select FEATURE GROUPS.
    Each original feature i forms a group: [i, n_original+i, ...]

    Args:
        gate: Gate values (abs)
        n_original: Number of original features (before expansion)
        k_original: Number of ORIGINAL features to select
        degree: Polynomial degree

    Returns:
        List of selected feature indices (expanded)
    """
    gate_abs = gate.abs()

    # Compute group importance: sum of original + squared
    # Group importance for feature i = gate[i] + gate[n_original + i]
    original_importance = gate_abs[:n_original]
    squared_importance = gate_abs[n_original:2*n_original] if degree >= 2 else torch.zeros(n_original)
    group_importance = original_importance + squared_importance

    # Select top-k_original groups
    top_groups = group_importance.topk(k_original).indices.tolist()

    # Return all expanded indices for selected groups
    selected = []
    for g in top_groups:
        selected.append(g)  # original feature
        if degree >= 2:
            selected.append(n_original + g)  # squared feature

    return selected


def run_single_fold(dataset, fold_idx, train_idx, test_idx, device, model_kwargs, C_schedule, epochs_per_phase, warmup_epochs, prune_threshold):
    """Run one fold."""
    X_train = dataset["X"][train_idx]
    y_train = dataset["y"][train_idx]
    X_test = dataset["X"][test_idx]
    y_test = dataset["y"][test_idx]

    selected, history, model = gradual_admm_with_pruning(
        model_class=GatedFeatureSelectionMLP,
        model_kwargs=model_kwargs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=dataset["n_classes"],
        ground_truth=dataset["ground_truth"],
        k=dataset["k"],
        C_schedule=C_schedule,
        prune_threshold=prune_threshold,
        prune_mode="soft",
        reweight=False,
        epochs_per_phase=epochs_per_phase,
        warmup_epochs=warmup_epochs,
        device=device,
        verbose=False,
    )

    # Use GROUP SELECTION instead of individual feature selection
    n_original = dataset["original_n_features"]

    # Determine k_original based on dataset
    # XOR: k_original=2 (no squared needed)
    # Ring: k_original=2 (needs squared)
    # Ring+XOR: k_original=4 (Ring needs 2, XOR needs 2)
    if dataset["name"] == "xor":
        k_original = 2
    elif dataset["name"] == "ring":
        k_original = 2
    elif dataset["name"] in ["ring+xor", "ring+xor+sum"]:
        k_original = 4

    # Apply group selection
    selected_group = group_selection_from_gate(model.gate, n_original, k_original, degree=2)

    # Compute best-k using group-selected features
    best_k = compute_best_k(selected_group, dataset["ground_truth"], dataset["k"])

    gate_tensor = model.gate.detach()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        X_gated = X_test_tensor * gate_tensor
        outputs = model(X_gated)
        if outputs.shape[1] == 1:
            y_pred_proba = torch.sigmoid(outputs).squeeze().cpu().numpy()
        else:
            probs = torch.softmax(outputs, dim=1)
            y_pred_proba = probs[:, 1].cpu().numpy()

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0.5

    return {"fold": fold_idx, "best_k": best_k, "auc": auc, "selected": selected_group, "selected_group": selected_group}


def run_benchmark():
    """Run benchmark with polynomial expansion + GROUP SELECTION."""
    print("=" * 70)
    print("BEST GRADUAL ADMM + POLYNOMIAL (degree=2) + GROUP SELECTION")
    print("Configuration: Soft mask + No re-weighting + Group selection")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    datasets = []
    for name in ["xor", "ring", "ring+xor", "ring+xor+sum"]:
        dataset = create_synthetic_dataset_with_poly(name, n_samples=1000, n_features=128, degree=2)
        datasets.append(dataset)

    model_kwargs = {
        "latent_size": 32,
        "n_hidden_layers": 2,
        "dropout": 0.043,
        "activation": "mish",
        "feat_drop": 0.6,
        "bounded_gate": False,
    }

    C_schedule = [0.1, 0.2, 0.3, 0.4, 0.5]
    epochs_per_phase = 100
    warmup_epochs = 100
    prune_threshold = 0.1

    n_folds = 6
    results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset['name']} (n_features={dataset['n_features']})")
        print(f"{'='*60}")

        fold_results = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset["X"], dataset["y"])):
            print(f"\nFold {fold_idx + 1}/{n_folds}...")

            fold_result = run_single_fold(
                dataset=dataset,
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                device=device,
                model_kwargs=model_kwargs,
                C_schedule=C_schedule,
                epochs_per_phase=epochs_per_phase,
                warmup_epochs=warmup_epochs,
                prune_threshold=prune_threshold,
            )
            fold_results.append(fold_result)
            print(f"  best-k: {fold_result['best_k']:.4f}, AUC: {fold_result['auc']:.4f}")

        best_k_values = [r["best_k"] for r in fold_results]
        auc_values = [r["auc"] for r in fold_results]

        results[dataset["name"]] = {
            "fold_results": fold_results,
            "mean_best_k": np.mean(best_k_values),
            "std_best_k": np.std(best_k_values),
            "mean_auc": np.mean(auc_values),
            "std_auc": np.std(auc_values),
            "k": dataset["k"],
            "n_features": dataset["n_features"],
        }

        print(f"\n{dataset['name']} Summary: best-k = {np.mean(best_k_values):.4f}, AUC = {np.mean(auc_values):.4f}")

    overall_mean_best_k = np.mean([results[ds]["mean_best_k"] for ds in results])
    overall_mean_auc = np.mean([results[ds]["mean_auc"] for ds in results])

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Overall Mean best-k: {overall_mean_best_k:.4f}")
    print(f"Overall Mean AUC: {overall_mean_auc:.4f}")

    # Comparison table
    print("\nComparison (degree=1 vs degree=2):")
    print("Dataset      | best-k (d1) | AUC (d1) | best-k (d2) | AUC (d2) | AUC improvement")
    print("-" * 70)
    for ds_name in ["xor", "ring", "ring+xor", "ring+xor+sum"]:
        d2_auc = results[ds_name]["mean_auc"]
        d2_bestk = results[ds_name]["mean_best_k"]
        # d1 values from previous run
        d1_values = {"xor": (1.00, 0.99), "ring": (1.00, 0.42), "ring+xor": (1.00, 0.61), "ring+xor+sum": (1.00, 0.57)}
        d1_bestk, d1_auc = d1_values[ds_name]
        improvement = d2_auc - d1_auc
        print(f"{ds_name:12} | {d1_bestk:.2f}       | {d1_auc:.2f}    | {d2_bestk:.2f}       | {d2_auc:.2f}    | {improvement:+.2f}")

    # Save results
    output_dir = PROJECT_ROOT / "results" / "main"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"best_gradual_poly2_benchmark_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "method": "gradual_soft_no_rw_poly2",
                "degree": 2,
                "n_samples": 1000,
                "n_folds": 6,
            },
            "results": results,
            "overall_mean_best_k": overall_mean_best_k,
            "overall_mean_auc": overall_mean_auc,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    results = run_benchmark()