# -*- coding: utf-8 -*-
"""
Iterative Run variants for feature selection.

Implements Lottery Ticket style iterative pruning:
- Variant A: Hard Iterative Pruning (with weight rewind)
- Variant B: Soft Iterative with Knowledge Distillation
- Variant C: Gradual ADMM Tightening (integrates with existing ADMM)
- Variant D: Iterative Gate Refinement
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Variant A: Hard Iterative Pruning (Classic Lottery Ticket)
# ---------------------------------------------------------------------------


def iterative_hard_pruning(
    model_class: Callable,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    train_fn: Callable,  # Training function (e.g., _train_input_group)
    score_fn: Callable,  # Function to get feature scores
    n_features_target: int,
    prune_ratio: float = 0.2,
    n_rounds: int = 10,
    rewind_to_init: bool = True,
    device: Optional[str] = None,
    verbose: bool = True,
) -> tuple[list[int], list[dict]]:
    """
    Classic Lottery Ticket style iterative pruning.

    Args:
        model_class: Class to instantiate model
        model_kwargs: kwargs for model constructor
        train_fn: Training function that trains model in-place
        score_fn: Function that returns feature importance scores
        n_features_target: Stop when features <= this number
        prune_ratio: Fraction of features to remove each round
        rewind_to_init: If True, reset weights to initialization each round
        n_rounds: Maximum number of pruning rounds

    Returns:
        final_features: List of remaining feature indices
        history: List of dicts with per-round stats
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = X_train.shape[1]
    current_features = list(range(n_features))
    history = []

    # Initial model
    model = model_class(input_size=n_features, n_classes=n_classes, **model_kwargs)
    model.to(device)

    # Save initial weights for rewinding
    if rewind_to_init:
        init_state = copy.deepcopy(model.state_dict())

    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n=== Round {round_idx + 1}/{n_rounds} ===")
            print(f"Features: {len(current_features)}")

        # Create model with current feature subset
        model = model_class(
            input_size=len(current_features), n_classes=n_classes, **model_kwargs
        )

        # Rewind or inherit weights
        if rewind_to_init and round_idx > 0:
            # Load matching subset of initial weights
            new_state = _subset_state_dict(init_state, current_features, n_features)
            _load_weights_subset(model, new_state)

        model.to(device)

        # Train
        train_fn(
            model,
            X_train[:, current_features],
            y_train,
            n_classes,
            device=device,
        )

        # Get feature scores
        scores = score_fn(model)
        scores = np.abs(scores.detach().cpu().numpy())

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test[:, current_features], dtype=torch.float32, device=device)
            y_pred = model(X_test_t)

        # Prune bottom-k%
        n_to_prune = max(1, int(len(current_features) * prune_ratio))
        if len(current_features) - n_to_prune < n_features_target:
            n_to_prune = len(current_features) - n_features_target

        keep_indices = np.argsort(scores)[n_to_prune:]
        prune_indices = np.argsort(scores)[:n_to_prune]

        # Record history
        round_stats = {
            "round": round_idx,
            "n_features": len(current_features),
            "n_pruned": n_to_prune,
            "scores": scores,
            "keep_indices": keep_indices.copy(),
        }
        history.append(round_stats)

        # Update feature set
        current_features = [current_features[i] for i in keep_indices]

        if verbose:
            print(f"Pruned {n_to_prune} features, {len(current_features)} remaining")

        if len(current_features) <= n_features_target:
            if verbose:
                print(f"Reached target: {len(current_features)} <= {n_features_target}")
            break

    return current_features, history


def _subset_state_dict(
    full_state: dict,
    feature_indices: list[int],
    original_n_features: int,
) -> dict:
    """Extract subset of weights corresponding to selected features."""
    new_state = {}
    n_selected = len(feature_indices)

    for name, param in full_state.items():
        # Gate parameter: select elements (1D)
        if "gate" in name and param.dim() == 1:
            if param.shape[0] == original_n_features:
                new_state[name] = param[feature_indices].clone()
            else:
                new_state[name] = param.clone()
        # First linear layer weight: select columns (2D)
        # Could be named "first_linear.weight" or "layers.X.weight" (where X varies based on dropout)
        elif param.dim() == 2 and param.shape[1] == original_n_features:
            new_state[name] = param[:, feature_indices].clone()
        # First linear layer bias or other 1D params that depend on original_n_features
        elif param.dim() == 1 and param.shape[0] == original_n_features:
            new_state[name] = param[feature_indices].clone()
        else:
            new_state[name] = param.clone()

    return new_state


def _load_weights_subset(model: nn.Module, state_dict: dict):
    """Load weights into model, handling shape mismatches by only loading matching shapes."""
    model_state = model.state_dict()

    for name, param in model_state.items():
        if name in state_dict:
            source_param = state_dict[name]
            if source_param.shape == param.shape:
                param.copy_(source_param)
            # else: shapes don't match, keep random initialization


# ---------------------------------------------------------------------------
# Variant B: Soft Iterative with Knowledge Distillation
# ---------------------------------------------------------------------------


def iterative_with_distillation(
    model_class: Callable,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    n_features_target: int,
    prune_ratio: float = 0.2,
    temperature: float = 4.0,
    alpha: float = 0.5,
    epochs_per_round: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[str] = None,
    verbose: bool = True,
) -> tuple[list[int], nn.Module]:
    """
    Iterative pruning with teacher-student distillation.

    Args:
        temperature: Softmax temperature for distillation
        alpha: Balance between task loss (alpha) and distillation loss (1-alpha)

    Returns:
        final_features: List of remaining feature indices
        best_model: Trained student model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = X_train.shape[1]
    current_features = list(range(n_features))

    # Train teacher with all features
    teacher = model_class(input_size=n_features, n_classes=n_classes, **model_kwargs)
    teacher.to(device)
    _train_standard(teacher, X_train, y_train, n_classes, epochs=epochs_per_round, device=device)
    teacher.eval()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    for round_idx in range(20):  # Max rounds
        if verbose:
            print(f"\n=== Round {round_idx + 1} ===")
            print(f"Features: {len(current_features)}")

        # Create student
        student = model_class(
            input_size=len(current_features), n_classes=n_classes, **model_kwargs
        )
        student.to(device)

        # Distillation training
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        criterion_task = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train[:, current_features], dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long if n_classes > 2 else torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs_per_round):
            student.train()
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = student(x_batch)

                # Task loss
                if n_classes > 2:
                    task_loss = criterion_task(logits, y_batch)
                else:
                    task_loss = criterion_task(logits.squeeze(), y_batch.float())

                # Distillation loss
                with torch.no_grad():
                    teacher_logits = teacher(
                        torch.cat([
                            torch.zeros(x_batch.shape[0], n_features, device=device),
                            x_batch,
                        ], dim=1) if False else _pad_features(
                            x_batch, current_features, n_features, device
                        )
                    )

                distill_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                loss = alpha * task_loss + (1 - alpha) * distill_loss
                loss.backward()
                optimizer.step()

        # Get scores and prune
        scores = _get_model_scores(student)
        n_to_prune = max(1, int(len(current_features) * prune_ratio))

        if len(current_features) - n_to_prune < n_features_target:
            n_to_prune = len(current_features) - n_features_target

        keep_indices = np.argsort(scores)[n_to_prune:]
        current_features = [current_features[i] for i in keep_indices]

        if verbose:
            print(f"Pruned {n_to_prune}, {len(current_features)} remaining")

        if len(current_features) <= n_features_target:
            break

        # Teacher = best student
        teacher.load_state_dict(student.state_dict())

    return current_features, student


def _pad_features(x, feature_indices, n_features, device):
    """Pad subset features to full dimension with zeros."""
    batch_size = x.shape[0]
    full = torch.zeros(batch_size, n_features, device=device, dtype=x.dtype)
    full[:, feature_indices] = x
    return full


def _train_standard(model, X, y, n_classes, epochs=100, lr=1e-3, device="cpu"):
    """Standard supervised training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long if n_classes > 2 else torch.float32, device=device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        if n_classes > 2:
            loss = criterion(out, y_t)
        else:
            loss = criterion(out.squeeze(), y_t.float())
        loss.backward()
        optimizer.step()


def _get_model_scores(model):
    """Get feature importance scores from model."""
    if hasattr(model, "get_feature_scores"):
        scores = model.get_feature_scores().detach().cpu().numpy()
    else:
        # Use first layer weights
        w = model.first_linear.weight.detach().cpu().numpy()
        scores = np.linalg.norm(w, axis=0)
    return np.abs(scores)


# ---------------------------------------------------------------------------
# Variant C: Gradual ADMM Tightening
# ---------------------------------------------------------------------------


def train_with_gradual_admm(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    initial_C: float = 0.001,
    final_C: float = 0.1,
    n_phases: int = 5,
    epochs_per_phase: int = 100,
    warmup_epochs: int = 50,
    device: Optional[str] = None,
    verbose: bool = True,
    **admm_kwargs,
) -> dict:
    """
    Gradually increase ADMM penalty to induce sparsity.

    No explicit feature removal - features naturally approach zero.
    Integrates with existing _train_input_group training loop.

    Args:
        initial_C: Starting sparsity coefficient (small = less sparsity)
        final_C: Final sparsity coefficient (large = more sparsity)
        n_phases: Number of phases to interpolate C
        epochs_per_phase: Epochs to train in each phase
    """
    from .admm_input_group_wrapper import _train_input_group

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    C_schedule = np.linspace(initial_C, final_C, n_phases)
    history = {"phases": [], "alive_features": []}

    for phase_idx, C in enumerate(C_schedule):
        if verbose:
            print(f"\n=== Phase {phase_idx + 1}/{n_phases}: C={C:.6f} ===")

        # Train with current C
        _train_input_group(
            model,
            X_train,
            y_train,
            n_classes,
            C=C,
            epochs=epochs_per_phase,
            warmup_epochs=warmup_epochs if phase_idx == 0 else 0,
            device=device,
            **admm_kwargs,
        )

        # Count alive features
        gate = model.get_gate_values() if hasattr(model, "get_gate_values") else model.gate
        alive = (torch.abs(gate) > 1e-4).sum().item()

        history["phases"].append(phase_idx)
        history["alive_features"].append(alive)

        if verbose:
            print(f"Alive features: {alive}")

    return history


# ---------------------------------------------------------------------------
# Variant D: Iterative Gate Refinement
# ---------------------------------------------------------------------------


def iterative_gate_refinement(
    model_class: Callable,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    train_fn: Callable,
    threshold: float = 0.01,
    n_iterations: int = 10,
    warmup_epochs: int = 50,
    admm_epochs: int = 100,
    min_features: int = 5,
    device: Optional[str] = None,
    verbose: bool = True,
) -> tuple[list[int], dict]:
    """
    Iterative gate refinement: train, remove small gates, continue.

    1. Train model with ADMM
    2. Find features with |gate| < threshold
    3. Remove those features
    4. Reinitialize and continue training
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = X_train.shape[1]
    alive_features = list(range(n_features))
    history = {"iterations": [], "n_removed": [], "alive_count": []}

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")
            print(f"Features: {len(alive_features)}")

        # Create model with current feature subset
        model = model_class(
            input_size=len(alive_features), n_classes=n_classes, **model_kwargs
        )
        model.to(device)

        # Train
        train_fn(
            model,
            X_train[:, alive_features],
            y_train,
            n_classes,
            warmup_epochs=warmup_epochs if iteration == 0 else 0,
            epochs=admm_epochs,
            device=device,
        )

        # Get gate values
        gate = model.get_gate_values() if hasattr(model, "get_gate_values") else model.gate
        gate_abs = torch.abs(gate).detach().cpu().numpy()

        # Find features to remove
        to_remove = gate_abs < threshold
        n_removed = to_remove.sum()

        history["iterations"].append(iteration)
        history["n_removed"].append(n_removed)
        history["alive_count"].append(len(alive_features))

        if verbose:
            print(f"Gate range: [{gate_abs.min():.6f}, {gate_abs.max():.6f}]")
            print(f"Removing {n_removed} features (|gate| < {threshold})")

        if n_removed == 0:
            if verbose:
                print("No features removed, stopping")
            break

        # Update alive features
        alive_features = [alive_features[i] for i in range(len(alive_features)) if not to_remove[i]]

        if len(alive_features) <= min_features:
            if verbose:
                print(f"Minimum features reached: {len(alive_features)}")
            break

    return alive_features, history