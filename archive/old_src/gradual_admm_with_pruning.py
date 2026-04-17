# -*- coding: utf-8 -*-
"""
Gradual ADMM with Pruning + Re-weighting.

Implements fair comparison between soft and hard pruning by:
1. Gradually increasing ADMM penalty C
2. Optionally pruning weak features (soft mask or hard delete)
3. Optionally re-weighting surviving gates to maintain energy

Variants:
- gradual_none: Only increase C, no pruning
- gradual_soft: Soft mask + re-weight
- gradual_soft_no_rw: Soft mask, no re-weight
- gradual_hard: Hard delete + re-weight
- gradual_hard_no_rw: Hard delete, no re-weight
"""

from __future__ import annotations

import copy
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Re-weighting Functions
# ---------------------------------------------------------------------------


def reweight_gates(
    gate: torch.Tensor,
    alive_mask: torch.Tensor,
    target_sum: float,
    mode: str = "scale",
) -> torch.Tensor:
    """
    Re-weight surviving gates to maintain total gate energy.

    Args:
        gate: Current gate values
        alive_mask: Boolean mask for alive features
        target_sum: Target sum of absolute gate values
        mode: Re-weighting mode
            - "scale": Simple scaling to target
            - "preserve_ratio": Preserve relative importance ratios
            - "gradual": Gradual scaling (cap at max_scale)

    Returns:
        new_gate: Re-weighted gate values
    """
    current_sum = (gate.abs() * alive_mask).sum().item()

    if current_sum < 1e-8:
        # All gates are zero, nothing to re-weight
        return gate * alive_mask

    if mode == "scale":
        scale = target_sum / current_sum
        new_gate = gate * alive_mask * scale

    elif mode == "preserve_ratio":
        # Preserve relative importance among alive features
        alive_gate = gate * alive_mask
        ratios = alive_gate.abs() / alive_gate.abs().sum()
        new_gate = ratios * target_sum * (alive_gate.sign())

    elif mode == "gradual":
        # Gradual scaling to avoid sudden jumps
        max_scale = 1.5  # Cap at 1.5x increase
        scale = min(max_scale, target_sum / current_sum)
        new_gate = gate * alive_mask * scale

    else:
        raise ValueError(f"Unknown reweight mode: {mode}")

    return new_gate


def compute_gate_energy(gate: torch.Tensor) -> Dict[str, float]:
    """Compute gate energy statistics."""
    gate_abs = gate.abs()
    return {
        "total_sum": gate_abs.sum().item(),
        "max_val": gate_abs.max().item(),
        "min_val": gate_abs.min().item(),
        "mean_val": gate_abs.mean().item(),
        "alive_count": (gate_abs > 1e-4).sum().item(),
    }


# ---------------------------------------------------------------------------
# Gradual ADMM Training with Pruning
# ---------------------------------------------------------------------------


def train_one_phase_admm(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    C: float,
    epochs: int = 50,
    warmup_epochs: int = 0,
    lr: float = 0.005,
    batch_size: int = 64,
    device: str = "cuda",
    use_ratio_norm: bool = True,
) -> Dict:
    """
    Train one phase with proper ADMM using existing _train_input_group.

    Uses the correct ADMM implementation from admm_input_group_wrapper.py.

    Args:
        epochs: Number of ADMM epochs AFTER warmup (not total epochs)
        warmup_epochs: Number of warmup epochs before ADMM
    """
    # Import the correct ADMM training function
    from .admm_input_group_wrapper import _train_input_group

    model.to(device)

    # Total epochs = warmup + ADMM epochs
    # _train_input_group expects epochs to be total, not just ADMM phase
    total_epochs = warmup_epochs + epochs

    # Call the proper ADMM training
    _train_input_group(
        model,
        X_train,
        y_train,
        n_classes,
        lr=lr,
        C=C,
        epochs=total_epochs,  # Total epochs (warmup + ADMM)
        warmup_epochs=warmup_epochs,  # Warmup phase length
        batch_size=batch_size,
        device=device,
        use_ratio_norm=use_ratio_norm,
        optimizer_type="adam",  # Adam for better convergence
    )

    # Collect gate statistics
    history = {"losses": [], "gate_stats": []}

    if hasattr(model, "gate"):
        gate_stats = compute_gate_energy(model.gate.detach())
        history["gate_stats"].append(gate_stats)

    return history


def gradual_admm_with_pruning(
    model_class: Callable,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    ground_truth: List[int],
    k: int,
    C_schedule: List[float] = [0.001, 0.005, 0.02, 0.05, 0.1],
    prune_threshold: float = 0.01,
    prune_mode: str = "soft",
    reweight: bool = True,
    reweight_mode: str = "scale",
    epochs_per_phase: int = 50,
    warmup_epochs: int = 20,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    """
    Gradual ADMM with optional pruning and re-weighting.

    Args:
        model_class: Class to instantiate model
        model_kwargs: kwargs for model constructor
        ground_truth: List of true relevant feature indices
        k: Number of features to select
        C_schedule: Gradually increasing penalty values
        prune_threshold: Threshold for pruning weak gates
        prune_mode: "none", "soft", or "hard"
        reweight: Whether to re-weight surviving gates
        reweight_mode: "scale", "preserve_ratio", or "gradual"

    Returns:
        selected_features: List of selected feature indices
        history: Dict with per-phase statistics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = X_train.shape[1]
    current_features = list(range(n_features))

    # Initialize model
    model = model_class(input_size=n_features, n_classes=n_classes, **model_kwargs)
    model.to(device)

    # Track initial gate energy for re-weighting
    if hasattr(model, "gate"):
        initial_gate_sum = model.gate.abs().sum().item()
    else:
        initial_gate_sum = 1.0

    history = {
        "phases": [],
        "prune_mode": prune_mode,
        "reweight": reweight,
        "initial_gate_sum": initial_gate_sum,
    }

    # Target for re-weighting: maintain small proportion of original energy
    # Lower target (0.2) prevents pushing gates back up after pruning
    target_ratio = 0.2  # Target 20% of original energy

    for phase_idx, C in enumerate(C_schedule):
        if verbose:
            print(f"\n=== Phase {phase_idx + 1}/{len(C_schedule)}: C={C:.4f} ===")
            print(f"Features: {len(current_features)}")

        # Phase 1: Train with current C
        # For hard pruning mode, X_train is already subsetted, so use it directly
        # For other modes, subset using current_features
        if prune_mode == "hard" and phase_idx > 0:
            X_train_phase = X_train  # Already subsetted in previous phase
            X_test_phase = X_test
        else:
            X_train_phase = X_train[:, current_features]
            X_test_phase = X_test[:, current_features]

        phase_history = train_one_phase_admm(
            model,
            X_train_phase,
            y_train,
            n_classes,
            C=C,
            epochs=epochs_per_phase,
            warmup_epochs=warmup_epochs if phase_idx == 0 else 0,
            device=device,
        )

        # Phase 2: Get gate values and identify weak features
        if hasattr(model, "gate"):
            gate = model.gate.detach().clone()
            gate_abs = gate.abs()

            alive_mask = gate_abs >= prune_threshold
            alive_indices = alive_mask.nonzero().squeeze().tolist()

            if isinstance(alive_indices, int):
                alive_indices = [alive_indices]

            gate_energy = compute_gate_energy(gate)

            if verbose:
                print(f"Gate energy: sum={gate_energy['total_sum']:.4f}, "
                      f"alive={gate_energy['alive_count']}")

        else:
            # No gate, use first layer weights
            w = model.first_linear.weight.detach()
            scores = w.abs().sum(dim=0)
            alive_mask = scores >= prune_threshold
            alive_indices = alive_mask.nonzero().squeeze().tolist()
            gate_energy = {"total_sum": scores.sum().item(), "alive_count": len(alive_indices)}

        # Phase 3: Prune + Re-weight (if enabled)
        if prune_mode != "none" and len(alive_indices) > 0:
            if prune_mode == "soft":
                # Soft: mask weak gates, keep all features
                if hasattr(model, "gate"):
                    model.gate.data = gate * alive_mask.float()

                    if reweight:
                        target_sum = initial_gate_sum * target_ratio
                        model.gate.data = reweight_gates(
                            model.gate.data,
                            alive_mask.float(),
                            target_sum,
                            mode=reweight_mode,
                        )

            elif prune_mode == "hard":
                # Hard: GRADUAL pruning - delete 10% per phase
                # More gentle than one-shot 90% deletion
                if hasattr(model, "gate"):
                    n_current = len(current_features)
                    # Delete 10% each phase, but keep at least k
                    n_keep = max(k, int(n_current * 0.9))  # Keep 90% (delete 10%)

                    gate_abs_sorted = gate_abs.sort(descending=True)
                    percentile_threshold = gate_abs_sorted.values[n_keep - 1].item()
                    hard_alive_mask = gate_abs >= percentile_threshold
                    hard_alive_indices = hard_alive_mask.nonzero().squeeze().tolist()

                    if isinstance(hard_alive_indices, int):
                        hard_alive_indices = [hard_alive_indices]

                    if len(hard_alive_indices) < k:
                        # Don't prune below k
                        if verbose:
                            print(f"Skipping hard prune: {len(hard_alive_indices)} < k={k}")
                    else:
                        # Map alive_indices (local) to original feature indices
                        original_alive = [current_features[i] for i in hard_alive_indices]

                        # Create new model with reduced input size
                        new_model = model_class(
                            input_size=len(hard_alive_indices),
                            n_classes=n_classes,
                            **model_kwargs
                        )
                        new_model.to(device)

                        # Copy weights (handle shape mismatch)
                        _copy_weights_subset(model, new_model, hard_alive_indices)

                        if hasattr(new_model, "gate") and reweight:
                            target_sum = initial_gate_sum * target_ratio
                            new_model.gate.data = reweight_gates(
                                new_model.gate.data,
                                torch.ones_like(new_model.gate.data),
                                target_sum,
                                mode=reweight_mode,
                            )

                        model = new_model

                        # Update data: subset to alive features
                        X_train = X_train[:, hard_alive_indices]
                        X_test = X_test[:, hard_alive_indices]

                        # Update current_features to track original indices for evaluation
                        current_features = original_alive

                        if verbose:
                            print(f"Gradual hard prune: {n_current} -> {len(hard_alive_indices)} features (deleted 10%)")

        # Compute best-k for current phase
        if hasattr(model, "gate"):
            gate_abs = model.gate.detach().abs()
            if prune_mode == "hard":
                # Map back to original indices
                top_k_local = gate_abs.topk(min(k, len(current_features))).indices.tolist()
                selected = [current_features[i] for i in top_k_local]
            else:
                top_k_indices = gate_abs.topk(k).indices.tolist()
                selected = top_k_indices
        else:
            selected = current_features[:k]

        best_k = len(set(selected) & set(ground_truth)) / k

        # Record phase results
        phase_result = {
            "phase": phase_idx,
            "C": C,
            "n_features": len(current_features),
            "alive_count": gate_energy["alive_count"],
            "gate_sum": gate_energy["total_sum"],
            "best_k": best_k,
            "selected": selected,
        }
        history["phases"].append(phase_result)

        if verbose:
            print(f"best-k: {best_k:.4f}")

    # Final selection
    if hasattr(model, "gate"):
        gate_abs = model.gate.detach().abs()
        if prune_mode == "hard":
            top_k_local = gate_abs.topk(min(k, len(current_features))).indices.tolist()
            selected_features = [current_features[i] for i in top_k_local]
        else:
            selected_features = gate_abs.topk(k).indices.tolist()
    else:
        selected_features = current_features[:k]

    return selected_features, history


def _copy_weights_subset(
    old_model: nn.Module,
    new_model: nn.Module,
    alive_indices: List[int],
):
    """Copy weights from old model to new model, handling shape mismatch."""
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    # Identify first linear layer by checking for input dimension match
    first_linear_name = None
    for name, param in old_state.items():
        if param.dim() == 2 and param.shape[1] == len(old_state.get('gate', torch.zeros(1))):
            first_linear_name = name
            break

    for name, param in new_state.items():
        if name in old_state:
            old_param = old_state[name]

            # Gate: subset of 1D tensor
            if "gate" in name and old_param.dim() == 1:
                param.copy_(old_param[alive_indices])

            # First linear layer: subset of columns (2D tensor)
            # Check by comparing input dimension to gate size
            elif name == first_linear_name and param.dim() == 2 and old_param.dim() == 2:
                # old_param shape: (out_features, in_features=old_gate_size)
                # new_param shape: (out_features, in_features=len(alive_indices))
                # Copy subset of columns corresponding to alive features
                param.copy_(old_param[:, alive_indices])

            # Other params: copy if shapes match
            elif old_param.shape == param.shape:
                param.copy_(old_param)


# ---------------------------------------------------------------------------
# Quick Run Function
# ---------------------------------------------------------------------------


def run_gradual_admm_ablation(
    datasets: List[Dict],
    model_class: Callable,
    model_kwargs: dict,
    variants: List[Dict] = None,
    seeds: List[int] = [42, 43, 44],
    device: str = "cuda",
    verbose: bool = True,
    # Training parameters
    C_schedule: List[float] = [0.01, 0.05, 0.1, 0.3, 0.5],
    epochs_per_phase: int = 50,
    warmup_epochs: int = 120,
    prune_threshold: float = 0.1,
) -> Dict:
    """
    Run ablation study on all datasets and variants.

    Args:
        datasets: List of dicts with 'name', 'X_train', 'y_train', 'X_test',
                  'y_test', 'n_classes', 'ground_truth', 'k'
        variants: List of dicts with 'name', 'prune_mode', 'reweight'
        seeds: Random seeds for reproducibility
        C_schedule: Gradually increasing penalty values
        epochs_per_phase: ADMM epochs per phase
        warmup_epochs: Warmup epochs before ADMM
        prune_threshold: Threshold for pruning weak gates

    Returns:
        results: Dict with all experiment results
    """
    if variants is None:
        variants = [
            {"name": "gradual_none", "prune_mode": "none", "reweight": False},
            {"name": "gradual_soft", "prune_mode": "soft", "reweight": True},
            {"name": "gradual_soft_no_rw", "prune_mode": "soft", "reweight": False},
            {"name": "gradual_hard", "prune_mode": "hard", "reweight": True},
            {"name": "gradual_hard_no_rw", "prune_mode": "hard", "reweight": False},
        ]

    results = {"variants": variants, "datasets": [], "runs": []}

    for dataset in datasets:
        dataset_name = dataset["name"]
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")

        for variant in variants:
            variant_name = variant["name"]
            if verbose:
                print(f"\n--- Variant: {variant_name} ---")

            run_results = []

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                selected, history = gradual_admm_with_pruning(
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    X_train=dataset["X_train"],
                    y_train=dataset["y_train"],
                    X_test=dataset["X_test"],
                    y_test=dataset["y_test"],
                    n_classes=dataset["n_classes"],
                    ground_truth=dataset["ground_truth"],
                    k=dataset["k"],
                    C_schedule=C_schedule,
                    prune_threshold=prune_threshold,
                    prune_mode=variant["prune_mode"],
                    reweight=variant["reweight"],
                    epochs_per_phase=epochs_per_phase,
                    warmup_epochs=warmup_epochs,
                    device=device,
                    verbose=verbose,
                )

                best_k = len(set(selected) & set(dataset["ground_truth"])) / dataset["k"]

                run_results.append({
                    "seed": seed,
                    "best_k": best_k,
                    "selected": selected,
                    "history": history,
                })

            # Aggregate
            mean_best_k = np.mean([r["best_k"] for r in run_results])
            std_best_k = np.std([r["best_k"] for r in run_results])

            results["runs"].append({
                "dataset": dataset_name,
                "variant": variant_name,
                "mean_best_k": mean_best_k,
                "std_best_k": std_best_k,
                "run_results": run_results,
            })

            if verbose:
                print(f"Result: best_k = {mean_best_k:.4f} ± {std_best_k:.4f}")

    # Summary table
    summary = {}
    for dataset in datasets:
        dataset_name = dataset["name"]
        summary[dataset_name] = {}
        for variant in variants:
            variant_name = variant["name"]
            for run in results["runs"]:
                if run["dataset"] == dataset_name and run["variant"] == variant_name:
                    summary[dataset_name][variant_name] = {
                        "mean": run["mean_best_k"],
                        "std": run["std_best_k"],
                    }

    results["summary"] = summary

    return results