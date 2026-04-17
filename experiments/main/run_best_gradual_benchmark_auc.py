# -*- coding: utf-8 -*-
"""
Run Full Benchmark with Best Gradual ADMM Configuration + Prediction AUC.

Uses the best configuration discovered in ablation:
- Gradual ADMM + Soft mask (no deletion) + No re-weighting

Protocol: m=128, n=1000, 6-fold CV (matching standard benchmark)
Datasets: XOR, Ring, Ring+XOR, Ring+XOR+Sum (4 datasets)

Outputs:
- best-k (feature recovery)
- AUC (prediction performance)
"""

import os
import sys
import json
import importlib
from datetime import datetime
from pathlib import Path

# Set up paths BEFORE any other imports
_script_path = Path(__file__).resolve() if '__file__' in dir() else Path.cwd()
PROJECT_ROOT = _script_path.parent.parent.parent.resolve()

# Force project root to be in sys.path at position 0
sys.path = [str(PROJECT_ROOT), str(PROJECT_ROOT / "Feature-Selection-Benchmark")] + sys.path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import using importlib to ensure correct module loading
src_gradual = importlib.import_module('src.gradual_admm_with_pruning')
gradual_admm_with_pruning = src_gradual.gradual_admm_with_pruning

src_admm = importlib.import_module('src.admm_input_group_wrapper')
GatedFeatureSelectionMLP = src_admm.GatedFeatureSelectionMLP
_Scaler = src_admm._Scaler

src_data = importlib.import_module('src.data')
generate_dataset = src_data.generate_dataset


def create_synthetic_dataset(name: str, n_samples: int = 1000, n_features: int = 128, seed: int = 42):
    """Create synthetic dataset matching benchmark protocol."""
    np.random.seed(seed)

    # Ground truth for each dataset type
    ground_truth_map = {
        "xor": [0, 1],  # k=2
        "ring": [0, 1],  # k=2
        "ring+xor": [0, 1, 2, 3],  # k=4
        "ring+xor+sum": [0, 1, 2, 3],  # k=4
    }

    # Generate dataset directly with supported name
    X, X_tilde, y = generate_dataset(name, n_samples=n_samples, n_features=n_features)

    ground_truth = ground_truth_map[name]
    k = len(ground_truth)

    # CRITICAL: Scale data
    scaler = _Scaler()
    X_scaled = scaler.fit_transform(X)

    n_classes = 2  # All synthetic datasets are binary

    return {
        "name": name,
        "X": X_scaled,
        "y": y,
        "n_classes": n_classes,
        "ground_truth": ground_truth,
        "k": k,
        "n_features": n_features,
        "n_samples": n_samples,
    }


def compute_best_k(selected: list, ground_truth: list, k: int) -> float:
    """Compute best-k metric: fraction of true features in top-k."""
    return len(set(selected) & set(ground_truth)) / k


def run_single_fold_with_auc(
    dataset: dict,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    model_kwargs: dict,
    C_schedule: list,
    epochs_per_phase: int,
    warmup_epochs: int,
    prune_threshold: float,
):
    """Run one fold of gradual ADMM and compute both best-k and AUC."""
    X_train = dataset["X"][train_idx]
    y_train = dataset["y"][train_idx]
    X_test = dataset["X"][test_idx]
    y_test = dataset["y"][test_idx]

    # Run gradual ADMM with soft mask + no re-weight (best config)
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
        prune_mode="soft",  # Best: soft mask
        reweight=False,  # Best: no re-weighting
        epochs_per_phase=epochs_per_phase,
        warmup_epochs=warmup_epochs,
        device=device,
        verbose=False,
    )

    # Compute best-k
    best_k = compute_best_k(selected, dataset["ground_truth"], dataset["k"])

    # Compute AUC from final model predictions
    gate_tensor = model.gate.detach()

    # Make predictions on test set
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        # Apply gate
        X_gated = X_test_tensor * gate_tensor
        outputs = model(X_gated)

        # Handle output shape - binary classification
        if outputs.shape[1] == 1:
            # Single output: use sigmoid
            y_pred_proba = torch.sigmoid(outputs).squeeze().cpu().numpy()
        else:
            # Multi-class: use softmax, take probability of class 1
            probs = torch.softmax(outputs, dim=1)
            y_pred_proba = probs[:, 1].cpu().numpy()

    # Compute AUC
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        # Handle edge case where all predictions are same class
        auc = 0.5

    return {
        "fold": fold_idx,
        "best_k": best_k,
        "auc": auc,
        "selected": selected,
        "history": history,
    }


def run_benchmark_with_auc():
    """Run full benchmark with best gradual configuration and compute AUC."""
    print("=" * 70)
    print("BEST GRADUAL ADMM BENCHMARK + AUC")
    print("Configuration: Soft mask + No re-weighting")
    print("Protocol: m=128, n=1000, 6-fold CV")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create datasets
    datasets = []
    for name in ["xor", "ring", "ring+xor", "ring+xor+sum"]:
        dataset = create_synthetic_dataset(name, n_samples=1000, n_features=128)
        datasets.append(dataset)
        print(f"Dataset {name}: n={dataset['n_samples']}, m={dataset['n_features']}, k={dataset['k']}")

    # Model configuration (matching benchmark)
    model_kwargs = {
        "latent_size": 32,
        "n_hidden_layers": 2,
        "dropout": 0.043,
        "activation": "mish",
        "feat_drop": 0.6,
        "bounded_gate": False,
    }

    # Best training configuration (from ablation)
    # 5 phases with gradually increasing C
    C_schedule = [0.1, 0.2, 0.3, 0.4, 0.5]
    epochs_per_phase = 100  # Total: 5*100 = 500 epochs + 100 warmup
    warmup_epochs = 100
    prune_threshold = 0.1  # Filter gates below 0.1

    # Cross-validation
    n_folds = 6

    results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset['name']}")
        print(f"{'='*60}")

        fold_results = []

        # Use StratifiedKFold for proper cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset["X"], dataset["y"])):
            print(f"\nFold {fold_idx + 1}/{n_folds}...")

            fold_result = run_single_fold_with_auc(
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

        # Compute statistics
        best_k_values = [r["best_k"] for r in fold_results]
        auc_values = [r["auc"] for r in fold_results if r["auc"] is not None]

        mean_best_k = np.mean(best_k_values)
        std_best_k = np.std(best_k_values)
        mean_auc = np.mean(auc_values) if auc_values else None
        std_auc = np.std(auc_values) if auc_values else None

        results[dataset["name"]] = {
            "fold_results": fold_results,
            "mean_best_k": mean_best_k,
            "std_best_k": std_best_k,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "k": dataset["k"],
        }

        auc_str = f"{mean_auc:.4f}" if mean_auc else "N/A"
        print(f"\n{dataset['name']} Summary: best-k = {mean_best_k:.4f} ± {std_best_k:.4f}, AUC = {auc_str}")

    # Compute overall mean
    overall_mean_best_k = np.mean([results[ds]["mean_best_k"] for ds in results.keys()])
    overall_mean_auc = np.mean([results[ds]["mean_auc"] for ds in results.keys() if results[ds]["mean_auc"]])

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Overall Mean best-k: {overall_mean_best_k:.4f}")
    print(f"Overall Mean AUC: {overall_mean_auc:.4f}")

    # Print comparison table
    print("\nTable 1 Comparison:")
    print("Method              | XOR    | Ring   | Ring+XOR | Ring+XOR+Sum | Mean best-k | Mean AUC")
    print("-" * 80)
    print(f"SADMM-FS            | {results['xor']['mean_best_k']:.2f}   | {results['ring']['mean_best_k']:.2f}   | {results['ring+xor']['mean_best_k']:.2f}     | {results['ring+xor+sum']['mean_best_k']:.2f}       | {overall_mean_best_k:.4f}       | {overall_mean_auc:.4f}")

    # Save results
    output_dir = PROJECT_ROOT / "results" / "main"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"best_gradual_benchmark_auc_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "method": "gradual_soft_no_rw",
                "n_samples": 1000,
                "n_features": 128,
                "n_folds": 6,
                "C_schedule": C_schedule,
                "epochs_per_phase": epochs_per_phase,
                "warmup_epochs": warmup_epochs,
                "prune_threshold": prune_threshold,
                "model_kwargs": model_kwargs,
            },
            "results": results,
            "overall_mean_best_k": overall_mean_best_k,
            "overall_mean_auc": overall_mean_auc,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = run_benchmark_with_auc()