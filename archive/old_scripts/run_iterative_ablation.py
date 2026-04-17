# -*- coding: utf-8 -*-
"""
Iterative Feature Selection Ablation (Lottery Ticket Style)

Run from within custom_admm directory:
    cd custom_admm && python run_iterative_ablation.py --quick

Compares single-pass ADMM vs iterative pruning methods.
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
from src.iterative_run import (
    iterative_hard_pruning,
    iterative_gate_refinement,
    train_with_gradual_admm,
)
from src.admm_input_group_wrapper import GatedFeatureSelectionMLP, _train_input_group, _Scaler


# ---------------------------------------------------------------------------
# Ground truth
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
# Fair training hyperparameters (matching main benchmark)
# ---------------------------------------------------------------------------

def _rho_for_dim(m: int) -> float:
    """Dimension-dependent rho value matching main benchmark."""
    if m < 64:
        return 20.0
    if m < 256:
        return 50.0
    if m < 512:
        return 100.0
    return 200.0


# Training hyperparameters matching main SADMM-FS method
FAIR_TRAINING_CONFIG = {
    "lr": 0.005,  # Main method default
    "C": 0.05,  # Ratio Norm sparsity coefficient
    "batch_size": 64,  # Main method default
    "optimizer_type": "adam",  # Main method uses Adam
    "use_ratio_norm": True,
    "use_admm": True,
    "use_early_stopping": False,  # Main method default
    "patience": 66,
    "val_split": 0.2,
    "epochs": 500,  # Main method default
    "warmup_epochs": 120,  # Main method default
}

# Model hyperparameters matching main benchmark
FAIR_MODEL_CONFIG = {
    "latent_size": 32,
    "n_hidden_layers": 2,
    "feat_drop": 0.6,
    "bounded_gate": False,
    "activation": "mish",
    "dropout": 0.043,
}


# ---------------------------------------------------------------------------
# Baseline: Single-pass ADMM
# ---------------------------------------------------------------------------

def run_single_pass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    n_classes: int,
    C: float = None,
    epochs: int = None,
    warmup_epochs: int = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, torch.nn.Module]:
    """
    Run single-pass ADMM feature selection with fair hyperparameters.
    """
    # Use fair hyperparameters if not specified
    if C is None:
        C = FAIR_TRAINING_CONFIG["C"]
    if epochs is None:
        epochs = FAIR_TRAINING_CONFIG["epochs"]
    if warmup_epochs is None:
        warmup_epochs = FAIR_TRAINING_CONFIG["warmup_epochs"]

    # Scale data (matching main benchmark)
    scaler = _Scaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = GatedFeatureSelectionMLP(
        input_size=n_features,
        n_classes=n_classes,
        latent_size=FAIR_MODEL_CONFIG["latent_size"],
        n_hidden_layers=FAIR_MODEL_CONFIG["n_hidden_layers"],
        feat_drop=FAIR_MODEL_CONFIG["feat_drop"],
        bounded_gate=FAIR_MODEL_CONFIG["bounded_gate"],
        activation=FAIR_MODEL_CONFIG["activation"],
        dropout=FAIR_MODEL_CONFIG["dropout"],
    )
    model.to(device)

    _train_input_group(
        model,
        X_train_scaled,
        y_train,
        n_classes,
        lr=FAIR_TRAINING_CONFIG["lr"],
        C=C,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        batch_size=FAIR_TRAINING_CONFIG["batch_size"],
        rho_init=_rho_for_dim(n_features),
        device=device,
        optimizer_type=FAIR_TRAINING_CONFIG["optimizer_type"],
        use_ratio_norm=FAIR_TRAINING_CONFIG["use_ratio_norm"],
        use_admm=FAIR_TRAINING_CONFIG["use_admm"],
        use_early_stopping=FAIR_TRAINING_CONFIG["use_early_stopping"],
        patience=FAIR_TRAINING_CONFIG["patience"],
        val_split=FAIR_TRAINING_CONFIG["val_split"],
    )

    scores = model.get_gate_values().detach().cpu().numpy()
    return scores, model


# ---------------------------------------------------------------------------
# Iterative methods
# ---------------------------------------------------------------------------

def run_iterative_hard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    n_classes: int,
    n_target: int,
    prune_ratio: float = 0.2,
    n_rounds: int = 5,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[int]]:
    """
    Run iterative hard pruning (NOT Lottery Ticket style - no weight rewinding).

    NOTE: Training epochs are normalized to match single-pass compute budget.
    Each round gets total_epochs / n_rounds epochs so total compute is comparable.
    """
    # Scale data (matching main benchmark)
    scaler = _Scaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # NORMALIZE EPOCHS: Divide total budget across rounds to match single-pass compute
    epochs_per_round = FAIR_TRAINING_CONFIG["epochs"] // n_rounds
    warmup_per_round = FAIR_TRAINING_CONFIG["warmup_epochs"] // n_rounds

    # Define model factory with fair hyperparameters
    def create_model(input_size, n_classes):
        return GatedFeatureSelectionMLP(
            input_size=input_size,
            n_classes=n_classes,
            latent_size=FAIR_MODEL_CONFIG["latent_size"],
            n_hidden_layers=FAIR_MODEL_CONFIG["n_hidden_layers"],
            feat_drop=FAIR_MODEL_CONFIG["feat_drop"],
            bounded_gate=FAIR_MODEL_CONFIG["bounded_gate"],
            activation=FAIR_MODEL_CONFIG["activation"],
            dropout=FAIR_MODEL_CONFIG["dropout"],
        )

    # Define training function with NORMALIZED epochs per round
    def train_fn(model, X, y, n_classes, device="cpu", **kwargs):
        _train_input_group(
            model, X, y, n_classes,
            lr=FAIR_TRAINING_CONFIG["lr"],
            C=FAIR_TRAINING_CONFIG["C"],
            epochs=epochs_per_round,  # NORMALIZED: total_epochs // n_rounds
            warmup_epochs=warmup_per_round,  # NORMALIZED
            batch_size=FAIR_TRAINING_CONFIG["batch_size"],
            rho_init=_rho_for_dim(X.shape[1]),
            device=device,
            optimizer_type=FAIR_TRAINING_CONFIG["optimizer_type"],
            use_ratio_norm=FAIR_TRAINING_CONFIG["use_ratio_norm"],
            use_admm=FAIR_TRAINING_CONFIG["use_admm"],
            use_early_stopping=FAIR_TRAINING_CONFIG["use_early_stopping"],
            patience=FAIR_TRAINING_CONFIG["patience"],
            val_split=FAIR_TRAINING_CONFIG["val_split"],
        )

    # Define score function
    def score_fn(model):
        return model.get_gate_values()

    final_features, history = iterative_hard_pruning(
        model_class=GatedFeatureSelectionMLP,
        model_kwargs={
            "latent_size": FAIR_MODEL_CONFIG["latent_size"],
            "n_hidden_layers": FAIR_MODEL_CONFIG["n_hidden_layers"],
            "feat_drop": FAIR_MODEL_CONFIG["feat_drop"],
            "bounded_gate": FAIR_MODEL_CONFIG["bounded_gate"],
            "activation": FAIR_MODEL_CONFIG["activation"],
            "dropout": FAIR_MODEL_CONFIG["dropout"],
        },
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_train_scaled,  # Dummy
        y_test=y_train,  # Dummy
        n_classes=n_classes,
        train_fn=train_fn,
        score_fn=score_fn,
        n_features_target=n_target,
        prune_ratio=prune_ratio,
        n_rounds=n_rounds,
        rewind_to_init=False,
        device=device,
        verbose=False,
    )

    # Create scores array
    scores = np.zeros(n_features)
    scores[final_features] = 1.0

    return scores, final_features


def run_lottery_ticket(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    n_classes: int,
    n_target: int,
    prune_ratio: float = 0.2,
    n_rounds: int = 5,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[int]]:
    """
    Run Lottery Ticket style iterative pruning WITH weight rewinding.

    Key difference from iterative_hard: weights are reset to initialization
    after each pruning step, as in the original Lottery Ticket Hypothesis paper.
    """
    scaler = _Scaler()
    X_train_scaled = scaler.fit_transform(X_train)

    epochs_per_round = FAIR_TRAINING_CONFIG["epochs"] // n_rounds
    warmup_per_round = FAIR_TRAINING_CONFIG["warmup_epochs"] // n_rounds

    def train_fn(model, X, y, n_classes, device="cpu", **kwargs):
        _train_input_group(
            model, X, y, n_classes,
            lr=FAIR_TRAINING_CONFIG["lr"],
            C=FAIR_TRAINING_CONFIG["C"],
            epochs=epochs_per_round,
            warmup_epochs=warmup_per_round,
            batch_size=FAIR_TRAINING_CONFIG["batch_size"],
            rho_init=_rho_for_dim(X.shape[1]),
            device=device,
            optimizer_type=FAIR_TRAINING_CONFIG["optimizer_type"],
            use_ratio_norm=FAIR_TRAINING_CONFIG["use_ratio_norm"],
            use_admm=FAIR_TRAINING_CONFIG["use_admm"],
            use_early_stopping=FAIR_TRAINING_CONFIG["use_early_stopping"],
            patience=FAIR_TRAINING_CONFIG["patience"],
            val_split=FAIR_TRAINING_CONFIG["val_split"],
        )

    def score_fn(model):
        return model.get_gate_values()

    final_features, history = iterative_hard_pruning(
        model_class=GatedFeatureSelectionMLP,
        model_kwargs={
            "latent_size": FAIR_MODEL_CONFIG["latent_size"],
            "n_hidden_layers": FAIR_MODEL_CONFIG["n_hidden_layers"],
            "feat_drop": FAIR_MODEL_CONFIG["feat_drop"],
            "bounded_gate": FAIR_MODEL_CONFIG["bounded_gate"],
            "activation": FAIR_MODEL_CONFIG["activation"],
            "dropout": FAIR_MODEL_CONFIG["dropout"],
        },
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_train_scaled,
        y_test=y_train,
        n_classes=n_classes,
        train_fn=train_fn,
        score_fn=score_fn,
        n_features_target=n_target,
        prune_ratio=prune_ratio,
        n_rounds=n_rounds,
        rewind_to_init=True,  # KEY DIFFERENCE: weight rewinding
        device=device,
        verbose=False,
    )

    scores = np.zeros(n_features)
    scores[final_features] = 1.0

    return scores, final_features


def run_gradual_admm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    n_classes: int,
    initial_C: float = 0.01,
    final_C: float = 0.1,
    n_phases: int = 5,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run gradual ADMM tightening with fair hyperparameters.
    """
    # Scale data (matching main benchmark)
    scaler = _Scaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = GatedFeatureSelectionMLP(
        input_size=n_features,
        n_classes=n_classes,
        latent_size=FAIR_MODEL_CONFIG["latent_size"],
        n_hidden_layers=FAIR_MODEL_CONFIG["n_hidden_layers"],
        feat_drop=FAIR_MODEL_CONFIG["feat_drop"],
        bounded_gate=FAIR_MODEL_CONFIG["bounded_gate"],
        activation=FAIR_MODEL_CONFIG["activation"],
        dropout=FAIR_MODEL_CONFIG["dropout"],
    )
    model.to(device)

    history = train_with_gradual_admm(
        model,
        X_train_scaled,
        y_train,
        n_classes,
        initial_C=initial_C,
        final_C=final_C,
        n_phases=n_phases,
        epochs_per_phase=FAIR_TRAINING_CONFIG["epochs"] // n_phases,
        warmup_epochs=FAIR_TRAINING_CONFIG["warmup_epochs"],
        device=device,
        verbose=False,
        # Additional fair training params
        lr=FAIR_TRAINING_CONFIG["lr"],
        batch_size=FAIR_TRAINING_CONFIG["batch_size"],
        rho_init=_rho_for_dim(n_features),
        optimizer_type=FAIR_TRAINING_CONFIG["optimizer_type"],
        use_ratio_norm=FAIR_TRAINING_CONFIG["use_ratio_norm"],
        use_admm=FAIR_TRAINING_CONFIG["use_admm"],
    )

    scores = model.get_gate_values().detach().cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_scores(
    scores: np.ndarray,
    ground_truth: List[int],
    k: int,
) -> Dict:
    """
    Evaluate feature recovery.

    Note: Uses np.abs(scores) to handle unbounded gates which can be negative.
    """
    # CRITICAL: Use absolute values for unbounded gates (can be negative)
    abs_scores = np.abs(scores)
    top_k = np.argsort(abs_scores)[-k:]
    top_2k = np.argsort(abs_scores)[-2*k:] if 2*k <= len(scores) else np.argsort(abs_scores)

    best_k = len(set(top_k) & set(ground_truth)) / k
    best_2k = len(set(top_2k) & set(ground_truth)) / min(2*k, len(ground_truth))

    return {
        "best_k": best_k,
        "best_2k": best_2k,
        "selected": top_k.tolist(),
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    dataset_name: str,
    n_samples: int,
    n_features: int,
    method: str,
    seed: int,
    device: str = "cpu",
) -> Dict:
    """
    Run a single experiment with fair hyperparameters.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate data
    X, X_tilde, y = generate_dataset(dataset_name, n_samples, n_features)

    # DATA CENTERING: Match main benchmark (transform from [0,1] to [-1,1])
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    # Split
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    n_classes = len(np.unique(y))
    ground_truth = GROUND_TRUTH[dataset_name]
    k = K_VALUES[dataset_name]

    # Run method with fair hyperparameters
    if method == "single_pass":
        scores, model = run_single_pass(
            X_train, y_train, n_features, n_classes,
            device=device
        )
    elif method == "iterative_hard":
        # DYNAMIC ROUNDS: Calculate rounds needed to reach target k
        # With prune_ratio=0.2, need ceil(log(k/n_features) / log(0.8)) rounds
        import math
        n_rounds_needed = max(5, math.ceil(math.log(k / n_features) / math.log(1 - 0.2)))
        scores, final_features = run_iterative_hard(
            X_train, y_train, n_features, n_classes,
            n_target=k, prune_ratio=0.2, n_rounds=n_rounds_needed, device=device
        )
    elif method == "lottery_ticket":
        # Lottery Ticket style WITH weight rewinding
        import math
        n_rounds_needed = max(5, math.ceil(math.log(k / n_features) / math.log(1 - 0.2)))
        scores, final_features = run_lottery_ticket(
            X_train, y_train, n_features, n_classes,
            n_target=k, prune_ratio=0.2, n_rounds=n_rounds_needed, device=device
        )
    elif method == "gradual_admm":
        scores = run_gradual_admm(
            X_train, y_train, n_features, n_classes,
            initial_C=0.01, final_C=0.1, n_phases=5, device=device
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Evaluate
    eval_results = evaluate_scores(scores, ground_truth, k)

    return {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        **eval_results,
    }


def run_ablation(
    quick: bool = False,
    seeds: Optional[List[int]] = None,
    device: str = "cpu",
) -> Dict:
    """
    Run full ablation comparing iterative vs single-pass methods.

    Uses protocol seed policy: 6-fold CV with seed = 42 + fold_idx
    """
    if seeds is None:
        # Protocol: 6-fold CV, seed = 42 + fold_idx
        seeds = [42, 43, 44] if quick else [42, 43, 44, 45, 46, 47]

    if quick:
        datasets = [("xor", 500, 32)]
    else:
        datasets = [
            ("xor", 1000, 128),
            ("ring", 1000, 128),
            ("ring+xor", 1000, 256),
        ]

    methods = ["single_pass", "iterative_hard", "lottery_ticket", "gradual_admm"]

    all_results = []

    for dataset_name, n_samples, n_features in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        for method in methods:
            print(f"\n  Method: {method}")

            for seed in seeds:
                try:
                    result = run_single_experiment(
                        dataset_name, n_samples, n_features, method, seed, device
                    )
                    all_results.append(result)
                    print(f"    Seed {seed}: best_k={result['best_k']:.4f}")
                except Exception as e:
                    print(f"    Seed {seed}: FAILED - {e}")

    # Aggregate
    summary = {}
    for dataset_name, _, _ in datasets:
        summary[dataset_name] = {}
        for method in methods:
            filtered = [r for r in all_results
                       if r["dataset"] == dataset_name and r["method"] == method]
            if filtered:
                summary[dataset_name][method] = {
                    "best_k_mean": np.mean([r["best_k"] for r in filtered]),
                    "best_k_std": np.std([r["best_k"] for r in filtered]),
                    "n_runs": len(filtered),
                }

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Iterative vs Single-Pass")
    print("=" * 60)
    for dataset_name, data in summary.items():
        print(f"\n{dataset_name}:")
        for method, stats in data.items():
            print(f"  {method:20s}: best_k = {stats['best_k_mean']:.4f} +/- {stats['best_k_std']:.4f}")

    return {
        "results": all_results,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("ITERATIVE FEATURE SELECTION ABLATION")
    print("=" * 60)

    results = run_ablation(quick=args.quick, device=args.device)

    # Save
    output_dir = os.path.join(ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"iterative_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()