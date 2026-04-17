# -*- coding: utf-8 -*-
"""
Run Experiment A: Gradual ADMM + Pruning + Re-weighting Ablation.

Compares 5 variants:
1. gradual_none: Only increase C, no pruning
2. gradual_soft: Soft mask + re-weight
3. gradual_soft_no_rw: Soft mask, no re-weight
4. gradual_hard: Hard delete + re-weight
5. gradual_hard_no_rw: Hard delete, no re-weight

Datasets: XOR, Ring, Ring+XOR (m=128)
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import generate_dataset
from src.gradual_admm_with_pruning import run_gradual_admm_ablation
from src.admm_input_group_wrapper import GatedFeatureSelectionMLP, _Scaler


def create_synthetic_dataset(name: str, n_samples: int = 500, n_features: int = 128, seed: int = 42):
    """Create synthetic dataset for experiment."""
    np.random.seed(seed)

    # Ground truth for each dataset type
    ground_truth_map = {
        "xor": [0, 1],
        "ring": [0, 1],
        "ring+xor": [0, 1, 2, 3],
    }

    # Generate dataset using existing data.py
    if name in ["xor", "ring", "ring+xor"]:
        X, X_tilde, y = generate_dataset(name, n_samples=n_samples, n_features=n_features)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    ground_truth = ground_truth_map[name]

    # CRITICAL: Scale data (matching quick test)
    scaler = _Scaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    n_train = int(0.8 * n_samples)
    X_train = X_scaled[:n_train]
    y_train = y[:n_train]
    X_test = X_scaled[n_train:]
    y_test = y[n_train:]

    n_classes = 2  # All synthetic datasets are binary

    return {
        "name": name,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "n_classes": n_classes,
        "ground_truth": ground_truth,
        "k": len(ground_truth),
    }


def main():
    """Run the ablation experiment."""
    print("=" * 60)
    print("Experiment A: Gradual ADMM + Pruning + Re-weighting")
    print("=" * 60)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seeds = [42, 43, 44]  # 3 seeds for statistical significance

    # Create datasets
    datasets = []
    for name in ["xor", "ring", "ring+xor"]:
        dataset = create_synthetic_dataset(name, n_samples=500, n_features=128, seed=42)
        datasets.append(dataset)
        print(f"Dataset {name}: n_train={len(dataset['X_train'])}, "
              f"m={dataset['X_train'].shape[1]}, k={dataset['k']}")

    # Fair model configuration (matching benchmark)
    model_kwargs = {
        "latent_size": 32,
        "n_hidden_layers": 2,
        "dropout": 0.043,  # Benchmark tuned value
        "activation": "mish",
        "feat_drop": 0.6,  # Benchmark tuned value
        "bounded_gate": False,
    }

    # Fair training configuration - MULTI PHASE for gradual pruning
    # 5 phases: each phase deletes 10% for hard pruning variants
    # 128 -> 115 -> 103 -> 93 -> 84 -> 75 (final ~60% of original)
    C_schedule = [0.1, 0.2, 0.3, 0.4, 0.5]  # Gradually increase C
    epochs_per_phase = 100  # ADMM epochs per phase
    warmup_epochs = 80  # Warmup before first ADMM phase

    # Variants to test
    variants = [
        {"name": "gradual_none", "prune_mode": "none", "reweight": False},
        {"name": "gradual_soft", "prune_mode": "soft", "reweight": True},
        {"name": "gradual_soft_no_rw", "prune_mode": "soft", "reweight": False},
        {"name": "gradual_hard", "prune_mode": "hard", "reweight": True},
        {"name": "gradual_hard_no_rw", "prune_mode": "hard", "reweight": False},
    ]

    # Run ablation with fair hyperparameters
    results = run_gradual_admm_ablation(
        datasets=datasets,
        model_class=GatedFeatureSelectionMLP,
        model_kwargs=model_kwargs,
        variants=variants,
        seeds=seeds,
        device=device,
        verbose=True,
        # Fair training parameters
        C_schedule=C_schedule,
        epochs_per_phase=epochs_per_phase,
        warmup_epochs=warmup_epochs,
        prune_threshold=0.1,  # Threshold to filter weak gates (higher = stricter)
    )

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    header = "Dataset | " + " | ".join([v["name"][:15] for v in variants])
    print(header)
    print("-" * len(header))

    for dataset_name in ["xor", "ring", "ring+xor"]:
        row = f"{dataset_name:10} |"
        for variant in variants:
            mean = results["summary"][dataset_name][variant["name"]]["mean"]
            std = results["summary"][dataset_name][variant["name"]]["std"]
            row += f" {mean:.2f}±{std:.2f} |"
        print(row)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_A_gradual_pruning_{timestamp}.json"

    # Convert to serializable format
    serializable_results = {
        "variants": variants,
        "seeds": seeds,
        "model_kwargs": model_kwargs,
        "summary": results["summary"],
        "runs": [],
    }

    for run in results["runs"]:
        serializable_run = {
            "dataset": run["dataset"],
            "variant": run["variant"],
            "mean_best_k": run["mean_best_k"],
            "std_best_k": run["std_best_k"],
            "run_results": [],
        }
        for r in run["run_results"]:
            serializable_run["run_results"].append({
                "seed": r["seed"],
                "best_k": r["best_k"],
                "selected": r["selected"],
            })
        serializable_results["runs"].append(serializable_run)

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()