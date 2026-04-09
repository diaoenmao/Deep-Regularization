# -*- coding: utf-8 -*-
"""
Transformer Pretrain Ablation

Run from within custom_admm directory:
    cd custom_admm && python run_transformer_pretrain_ablation.py --quick

Tests whether MAE-style pretraining can rescue transformer backbone for FS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Set up paths for running from custom_admm directory
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)  # custom_admm
sys.path.insert(0, os.path.join(ROOT, "src"))  # custom_admm/src
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "Feature-Selection-Benchmark", "src"))

# Now imports should work
from src.data import generate_dataset
from src.transformer_pretrain import (
    MaskedFeaturePretrainer,
    PretrainedTransformerWithGate,
    pretrain_transformer,
    finetune_with_admm,
    run_transformer_pretrain_experiment,
    get_recommended_config,
)
# Import ADMM utilities from sibling module
from src.admm_input_group_wrapper import GatedFeatureSelectionMLP, _train_input_group, _Scaler
from src.mentor_models import GatedTokenTransformerFS


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


FAIR_TRAINING_CONFIG = {
    "lr": 0.005,  # Main method default
    "C": 0.05,
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

FAIR_MODEL_CONFIG = {
    "latent_size": 32,
    "n_hidden_layers": 2,
    "feat_drop": 0.6,
    "bounded_gate": False,
    "activation": "mish",
    "dropout": 0.043,
}


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

# Transformer architecture matching main benchmark (backbone_tier2_synthetic_full.py)
TRANSFORMER_CONFIG = {
    "d_model": 32,  # Match main benchmark
    "n_heads": 4,
    "n_layers": 2,  # Match main benchmark
    "ff_dim": 128,
    "feat_drop": 0.6,
    "bounded_gate": False,
    "dropout": 0.1,
}


def run_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    n_classes: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run MLP baseline (SADMM-FS) with fair hyperparameters.
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

    _train_input_group(
        model, X_train_scaled, y_train, n_classes,
        lr=FAIR_TRAINING_CONFIG["lr"],
        C=FAIR_TRAINING_CONFIG["C"],
        epochs=FAIR_TRAINING_CONFIG["epochs"],
        warmup_epochs=FAIR_TRAINING_CONFIG["warmup_epochs"],
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

    return model.get_gate_values().detach().cpu().numpy()  # GatedFeatureSelectionMLP


def run_transformer_no_pretrain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    n_classes: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run transformer without pretraining using GatedTokenTransformerFS.
    Uses architecture matching main benchmark (d_model=32, n_layers=2).
    """
    # Scale data (matching main benchmark)
    scaler = _Scaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use GatedTokenTransformerFS with architecture matching main benchmark
    model = GatedTokenTransformerFS(
        input_size=n_features,
        n_classes=n_classes,
        d_model=TRANSFORMER_CONFIG["d_model"],
        n_heads=TRANSFORMER_CONFIG["n_heads"],
        n_layers=TRANSFORMER_CONFIG["n_layers"],
        ff_dim=TRANSFORMER_CONFIG["ff_dim"],
        feat_drop=TRANSFORMER_CONFIG["feat_drop"],
        bounded_gate=TRANSFORMER_CONFIG["bounded_gate"],
        dropout=TRANSFORMER_CONFIG["dropout"],
    )
    model.to(device)

    # Train with ADMM + uniform_penalty (matching main benchmark for transformers)
    _train_input_group(
        model, X_train_scaled, y_train, n_classes,
        lr=FAIR_TRAINING_CONFIG["lr"],
        C=FAIR_TRAINING_CONFIG["C"],
        epochs=FAIR_TRAINING_CONFIG["epochs"],
        warmup_epochs=FAIR_TRAINING_CONFIG["warmup_epochs"],
        batch_size=FAIR_TRAINING_CONFIG["batch_size"],
        rho_init=_rho_for_dim(n_features),
        device=device,
        optimizer_type=FAIR_TRAINING_CONFIG["optimizer_type"],
        use_ratio_norm=FAIR_TRAINING_CONFIG["use_ratio_norm"],
        use_admm=FAIR_TRAINING_CONFIG["use_admm"],
        use_early_stopping=FAIR_TRAINING_CONFIG["use_early_stopping"],
        patience=FAIR_TRAINING_CONFIG["patience"],
        val_split=FAIR_TRAINING_CONFIG["val_split"],
        uniform_penalty=True,  # CRITICAL: Match main benchmark for gate-based scores
    )

    return model.gate_values().detach().cpu().numpy()


def run_transformer_with_pretrain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    n_classes: int,
    pretrain_epochs: int = 24,  # Match main benchmark
    finetune_epochs: int = None,
    pretrain_lr: float = 1e-3,  # Better convergence (tested: 1e-5 -> 1e-3)
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Run transformer with masked feature pretraining.
    Uses GatedTokenTransformerFS with architecture matching main benchmark.

    Returns:
        scores: Feature importance scores
        pretrain_info: Dict with pretrain loss history
    """
    if finetune_epochs is None:
        finetune_epochs = FAIR_TRAINING_CONFIG["epochs"]

    # Scale data (matching protocol)
    scaler = _Scaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use GatedTokenTransformerFS with architecture matching main benchmark
    model = GatedTokenTransformerFS(
        input_size=n_features,
        n_classes=n_classes,
        d_model=TRANSFORMER_CONFIG["d_model"],
        n_heads=TRANSFORMER_CONFIG["n_heads"],
        n_layers=TRANSFORMER_CONFIG["n_layers"],
        ff_dim=TRANSFORMER_CONFIG["ff_dim"],
        feat_drop=TRANSFORMER_CONFIG["feat_drop"],
        bounded_gate=TRANSFORMER_CONFIG["bounded_gate"],
        dropout=TRANSFORMER_CONFIG["dropout"],
    )
    model.to(device)

    # Phase 1: Pretrain with masked reconstruction
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr, weight_decay=1e-4)
    X_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=FAIR_TRAINING_CONFIG["batch_size"],  # Protocol batch_size
        shuffle=True,
    )
    mask_prob = 0.15  # Match main benchmark

    # Loss logging for pretrain phase
    pretrain_losses = []

    for epoch in range(pretrain_epochs):
        epoch_losses = []
        for (x_batch,) in loader:
            mask = torch.rand_like(x_batch) < mask_prob
            x_masked = x_batch.clone()
            x_masked[mask] = 0.0
            recon = model.reconstruct_masked(x_masked) if hasattr(model, 'reconstruct_masked') else x_masked
            if mask.any():
                loss = ((recon - x_batch) ** 2)[mask].mean()
            else:
                loss = ((recon - x_batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        pretrain_losses.append(avg_loss)
        if verbose and epoch % 5 == 0:
            print(f"Pretrain epoch {epoch}: loss={avg_loss:.6f}")

    pretrain_info = {
        "final_loss": pretrain_losses[-1] if pretrain_losses else None,
        "loss_history": pretrain_losses,
        "pretrain_epochs": pretrain_epochs,
        "pretrain_lr": pretrain_lr,
    }

    # Phase 2: Fine-tune with ADMM (matching protocol)
    _train_input_group(
        model, X_train_scaled, y_train, n_classes,
        lr=FAIR_TRAINING_CONFIG["lr"],
        C=FAIR_TRAINING_CONFIG["C"],
        epochs=finetune_epochs,
        warmup_epochs=FAIR_TRAINING_CONFIG["warmup_epochs"],
        batch_size=FAIR_TRAINING_CONFIG["batch_size"],
        rho_init=_rho_for_dim(n_features),
        device=device,
        optimizer_type=FAIR_TRAINING_CONFIG["optimizer_type"],
        use_ratio_norm=FAIR_TRAINING_CONFIG["use_ratio_norm"],
        use_admm=FAIR_TRAINING_CONFIG["use_admm"],
        use_early_stopping=FAIR_TRAINING_CONFIG["use_early_stopping"],
        patience=FAIR_TRAINING_CONFIG["patience"],
        val_split=FAIR_TRAINING_CONFIG["val_split"],
        uniform_penalty=True,  # CRITICAL: Match main benchmark for gate-based scores
    )

    return model.gate_values().detach().cpu().numpy(), pretrain_info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_scores(scores: np.ndarray, ground_truth: List[int], k: int) -> Dict:
    # CRITICAL: Use absolute values for unbounded gates (can be negative)
    abs_scores = np.abs(scores)
    top_k = np.argsort(abs_scores)[-k:]
    best_k = len(set(top_k) & set(ground_truth)) / k
    return {"best_k": best_k, "selected": top_k.tolist()}


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
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, _, y = generate_dataset(dataset_name, n_samples, n_features)

    # DATA CENTERING: Match main benchmark (transform from [0,1] to [-1,1])
    X = 2.0 * X - 1.0

    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    n_classes = len(np.unique(y))
    ground_truth = GROUND_TRUTH[dataset_name]
    k = K_VALUES[dataset_name]

    if method == "mlp_baseline":
        scores = run_mlp_baseline(X_train, y_train, X_test, n_features, n_classes, device)
    elif method == "transformer_no_pretrain":
        scores = run_transformer_no_pretrain(X_train, y_train, X_test, n_features, n_classes, device)
    elif method == "transformer_pretrain":
        scores, pretrain_info = run_transformer_with_pretrain(X_train, y_train, X_test, n_features, n_classes, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")

    eval_results = evaluate_scores(scores, ground_truth, k)

    result = {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        **eval_results,
    }

    # Add pretrain info if available
    if method == "transformer_pretrain" and pretrain_info:
        result["pretrain_info"] = pretrain_info

    return result


def run_ablation(quick: bool = False, seeds: Optional[List[int]] = None, device: str = "cpu") -> Dict:
    if seeds is None:
        # Protocol: 6-fold CV, seed = 42 + fold_idx
        seeds = [42, 43, 44] if quick else [42, 43, 44, 45, 46, 47]

    if quick:
        datasets = [("xor", 500, 32)]
        methods = ["mlp_baseline", "transformer_pretrain"]
    else:
        datasets = [("xor", 1000, 128), ("ring", 1000, 128)]
        methods = ["mlp_baseline", "transformer_no_pretrain", "transformer_pretrain"]

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
                    import traceback
                    traceback.print_exc()

    # Aggregate
    summary = {}
    for dataset_name, _, _ in datasets:
        summary[dataset_name] = {}
        for method in methods:
            filtered = [r for r in all_results if r["dataset"] == dataset_name and r["method"] == method]
            if filtered:
                summary[dataset_name][method] = {
                    "best_k_mean": np.mean([r["best_k"] for r in filtered]),
                    "best_k_std": np.std([r["best_k"] for r in filtered]),
                    "n_runs": len(filtered),
                }

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Transformer Pretrain vs Baselines")
    print("=" * 60)
    for dataset_name, data in summary.items():
        print(f"\n{dataset_name}:")
        for method, stats in data.items():
            print(f"  {method:25s}: best_k = {stats['best_k_mean']:.4f} +/- {stats['best_k_std']:.4f}")

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
    print("TRANSFORMER PRETRAIN ABLATION")
    print("=" * 60)
    print(f"Device: {args.device}")

    results = run_ablation(quick=args.quick, device=args.device)

    output_dir = os.path.join(ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"transformer_pretrain_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()