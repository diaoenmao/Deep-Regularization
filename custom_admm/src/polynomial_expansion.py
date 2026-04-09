# -*- coding: utf-8 -*-
"""
Polynomial Feature Expansion + Selection variants.

Expansion:
- Variant A: sklearn PolynomialFeatures (degree=2/3)
- Variant B: Learned Expansion (current implementation)
- Variant C: Fourier Features

Selection:
- Variant A: Original-Feature Selection (current)
- Variant B: Expanded-Space Selection
- Variant C: Group Lasso on Expanded Features
- Variant D: Hierarchical Selection
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Expansion Variant A: sklearn PolynomialFeatures
# ---------------------------------------------------------------------------


def compute_polynomial_expansion(
    X: np.ndarray,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
) -> Tuple[np.ndarray, list[str], object]:
    """
    Expand features using polynomial combinations.

    Args:
        X: Input array of shape (n_samples, n_features)
        degree: Maximum polynomial degree
        interaction_only: If True, no x^2 terms, only x_i * x_j
        include_bias: If True, include constant term

    Returns:
        X_expanded: Expanded features
        feature_names: Names of expanded features
        poly: Fitted PolynomialFeatures object

    Example:
        degree=2: [x1, x2] -> [x1, x2, x1², x1*x2, x2²]
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
    )
    X_expanded = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out()

    return X_expanded, list(feature_names), poly


def compute_expanded_size(n_features: int, degree: int) -> int:
    """Compute number of features after polynomial expansion."""
    from math import comb
    # Sum of combinations: C(n+d, d) for d=1..degree
    total = sum(comb(n_features + d, d) for d in range(1, degree + 1))
    return int(total)


def get_expansion_groups(
    n_features: int,
    degree: int,
    feature_names: list[str],
) -> dict[int, list[int]]:
    """
    Map original feature indices to their expanded feature indices.

    Returns:
        groups: {original_idx: [expanded_idx1, expanded_idx2, ...]}

    Example:
        n_features=2, degree=2
        feature_names = ['x0', 'x1', 'x0²', 'x0 x1', 'x1²']
        groups = {0: [0, 2, 3], 1: [1, 3, 4]}
    """
    groups = {i: [] for i in range(n_features)}

    for exp_idx, name in enumerate(feature_names):
        # Parse feature name to find which original features are involved
        # Examples: 'x0', 'x1', 'x0²', 'x0 x1', 'x1²'
        parts = name.replace(' ', ' * ').replace('²', '^2').split(' * ')

        for part in parts:
            # Extract original feature index
            if part.startswith('x'):
                base = part.split('^')[0]  # Remove power if present
                orig_idx = int(base[1:])
                if orig_idx not in groups[orig_idx]:
                    pass  # Already in dict
                groups[orig_idx].append(exp_idx)

    # Remove duplicates
    for k in groups:
        groups[k] = list(set(groups[k]))

    return groups


# ---------------------------------------------------------------------------
# Expansion Variant C: Fourier Features
# ---------------------------------------------------------------------------


def compute_fourier_features(
    X: torch.Tensor,
    n_bands: int = 10,
    max_freq: float = 1.0,
) -> torch.Tensor:
    """
    Fourier feature encoding for continuous features.

    sin/cos encoding captures periodic patterns.

    Args:
        X: Input tensor of shape (batch, n_features)
        n_bands: Number of frequency bands
        max_freq: Maximum frequency

    Returns:
        X_fourier: (batch, n_features + 2 * n_bands * n_features)
    """
    bands = torch.linspace(0, max_freq, n_bands, device=X.device, dtype=X.dtype)

    features = [X]
    for b in bands:
        features.append(torch.sin(2 * np.pi * b * X))
        features.append(torch.cos(2 * np.pi * b * X))

    return torch.cat(features, dim=-1)


class FourierFeatureLayer(nn.Module):
    """Module wrapper for Fourier feature computation."""

    def __init__(self, n_bands: int = 10, max_freq: float = 1.0):
        super().__init__()
        self.n_bands = n_bands
        self.max_freq = max_freq
        # Register bands as buffer for device handling
        self.register_buffer(
            "bands",
            torch.linspace(0, max_freq, n_bands),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for b in self.bands:
            features.append(torch.sin(2 * np.pi * b * x))
            features.append(torch.cos(2 * np.pi * b * x))
        return torch.cat(features, dim=-1)

    def output_size(self, input_size: int) -> int:
        return input_size + 2 * self.n_bands * input_size


# ---------------------------------------------------------------------------
# Polynomial Feature Selection Model
# ---------------------------------------------------------------------------


class PolynomialFeatureSelectionModel(nn.Module):
    """
    Polynomial expansion + feature selection model.

    Supports multiple selection modes:
    - "expanded": Independent gates for each expanded feature
    - "group": Group lasso style, all expansions of a feature selected together
    - "hierarchical": Two-level selection (original -> expanded)
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        degree: int = 2,
        hidden_dims: list = [32, 32],
        selection_mode: str = "group",  # "expanded", "group", "hierarchical"
        activation: str = "mish",
        dropout: float = 0.1,
        bounded_gate: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.degree = degree
        self.n_classes = n_classes
        self.selection_mode = selection_mode
        self.bounded_gate = bounded_gate

        # Create polynomial transformer to get actual expanded size
        dummy_X = np.zeros((1, input_size))
        _, self.feature_names, self.poly = compute_polynomial_expansion(
            dummy_X, degree=degree
        )
        self.expanded_size = len(self.feature_names)

        # Build expansion groups
        self.expansion_groups = get_expansion_groups(input_size, degree, self.feature_names)

        # Gates
        if selection_mode == "expanded":
            # One gate per expanded feature
            self.gate = nn.Parameter(torch.zeros(self.expanded_size))
        elif selection_mode == "group":
            # One gate per original feature (shared for all its expansions)
            self.gate = nn.Parameter(torch.zeros(input_size))
        elif selection_mode == "hierarchical":
            # Two-level gates
            self.gate_original = nn.Parameter(torch.zeros(input_size))
            self.gate_expanded = nn.Parameter(torch.zeros(self.expanded_size))
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        # Predictor MLP
        act_fn = {"mish": nn.Mish, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        layers = []
        in_dim = self.expanded_size
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1 if n_classes <= 2 else n_classes))
        self.predictor = nn.Sequential(*layers)
        # Alias for compatibility with _train_input_group (expects model.layers)
        self.layers = self.predictor
        # Reference to first linear layer for score computation
        self.first_linear = self.predictor[0]

    def get_gate_scores(self) -> torch.Tensor:
        """
        Compute importance scores aligned with gate dimension.

        For "expanded" mode: returns ||W[0]||_2 for each expanded feature.
        For "group" mode: aggregates expanded scores to original feature scores.

        Returns:
            scores: Shape (gate_dim,) where gate_dim = input_size for group mode,
                    or expanded_size for expanded mode.
        """
        # Get weight norms from first linear layer (expanded space)
        weight = self.first_linear.weight  # shape (hidden, expanded_size)
        expanded_scores = torch.norm(weight, p=2, dim=0)  # shape (expanded_size,)

        if self.selection_mode == "expanded":
            return expanded_scores
        elif self.selection_mode == "group":
            # Aggregate expanded scores to original feature scores
            # Each original feature maps to multiple expanded features
            original_scores = torch.zeros(self.input_size, device=weight.device)
            for orig_idx, exp_indices in self.expansion_groups.items():
                # Use mean of expanded scores for this original feature
                original_scores[orig_idx] = expanded_scores[exp_indices].mean()
            return original_scores
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

    def gate_values(self) -> torch.Tensor:
        """Get effective gate values for each expanded feature."""
        if self.selection_mode == "expanded":
            g = self.gate
        elif self.selection_mode == "group":
            # Map original gates to expanded features
            g_expanded = torch.zeros(self.expanded_size, device=self.gate.device)
            for orig_idx, exp_indices in self.expansion_groups.items():
                g_expanded[exp_indices] = self.gate[orig_idx]
            g = g_expanded
        elif self.selection_mode == "hierarchical":
            # Product of original and expanded gates
            g_original_expanded = torch.zeros(self.expanded_size, device=self.gate_original.device)
            for orig_idx, exp_indices in self.expansion_groups.items():
                g_original_expanded[exp_indices] = self.gate_original[orig_idx]
            g = g_original_expanded * self.gate_expanded
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

        if self.bounded_gate:
            g = torch.sigmoid(g)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, input_size) raw features

        Returns:
            logits: (batch, n_classes)

        NOTE: This implementation breaks gradient flow through the polynomial
        expansion. The sklearn transform is non-differentiable, so:
        1. Gates can only learn through the downstream predictor
        2. No gradient signal informs gates about polynomial structure
        3. This is a fundamental limitation of the current design

        For a differentiable polynomial layer, consider implementing polynomial
        features in pure PyTorch (e.g., using torch.matmul for interactions).
        """
        # Expand features using sklearn transformer
        # WARNING: This breaks gradient flow - x_np has no gradients
        x_np = x.detach().cpu().numpy()
        x_exp = self.poly.transform(x_np)
        x_exp = torch.tensor(x_exp, dtype=x.dtype, device=x.device)

        # Apply gates
        g = self.gate_values()
        x_gated = x_exp * g

        # Predict
        return self.predictor(x_gated)

    def get_feature_scores(self) -> torch.Tensor:
        """
        Return feature importance scores aligned with gate dimension.

        For ADMM RatioNorm, scores must match gate shape:
        - "expanded" mode: returns weight norms for each expanded feature
        - "group" mode: returns aggregated weight norms for each original feature

        Returns:
            scores: Shape (gate_dim,) aligned with model.gate
        """
        # Use weight norms from first linear layer (importance-based scoring)
        weight = self.first_linear.weight  # shape (hidden, expanded_size)
        expanded_scores = torch.norm(weight, p=2, dim=0)  # shape (expanded_size,)

        if self.selection_mode == "expanded":
            # Gate dimension matches expanded dimension
            return expanded_scores
        elif self.selection_mode == "group":
            # Aggregate to original feature dimension (gate is per-original-feature)
            original_scores = torch.zeros(self.input_size, device=weight.device)
            for orig_idx, exp_indices in self.expansion_groups.items():
                # Use max score among expanded features for this original feature
                original_scores[orig_idx] = expanded_scores[exp_indices].max()
            return original_scores
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

    def get_original_feature_scores(self) -> torch.Tensor:
        """Return scores for original features (not expanded)."""
        if self.selection_mode == "expanded":
            # Aggregate expanded scores to original
            scores = torch.zeros(self.input_size, device=self.gate.device)
            for orig_idx, exp_indices in self.expansion_groups.items():
                scores[orig_idx] = self.gate_values()[exp_indices].abs().mean()
            return scores
        elif self.selection_mode == "group":
            return torch.abs(self.gate)
        elif self.selection_mode == "hierarchical":
            return torch.abs(self.gate_original)
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")


# ---------------------------------------------------------------------------
# Group Lasso Loss for Expanded Features
# ---------------------------------------------------------------------------


def group_lasso_loss(
    gate: torch.Tensor,
    expansion_groups: dict[int, list[int]],
) -> torch.Tensor:
    """
    Compute group lasso penalty for expanded features.

    Groups are defined by original features: all expansions of feature j
    are in one group.

    Args:
        gate: Gate values for expanded features, shape (expanded_size,)
        expansion_groups: {original_idx: [expanded_idx1, ...]}

    Returns:
        penalty: Sum of L2 norms of gate values in each group
    """
    penalty = 0.0
    for orig_idx, exp_indices in expansion_groups.items():
        group_gates = gate[exp_indices]
        penalty += torch.norm(group_gates, p=2)
    return penalty


# ---------------------------------------------------------------------------
# Hierarchical Selection
# ---------------------------------------------------------------------------


class HierarchicalFeatureSelector(nn.Module):
    """
    Two-level hierarchical feature selection.

    Level 1: Select k_original original features
    Level 2: Within each selected, select k_expanded expanded features
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        degree: int = 2,
        k_original: int = 10,
        k_expanded_per_feature: int = 2,
        hidden_dims: list = [32, 32],
    ):
        super().__init__()
        self.input_size = input_size
        self.degree = degree
        self.k_original = k_original
        self.k_expanded_per_feature = k_expanded_per_feature

        # Expansion info
        self.expanded_size = compute_expanded_size(input_size, degree)
        dummy_X = np.zeros((1, input_size))
        _, self.feature_names, self.poly = compute_polynomial_expansion(dummy_X, degree=degree)
        self.expansion_groups = get_expansion_groups(input_size, degree, self.feature_names)

        # Level 1 gates (original features)
        self.gate_original = nn.Parameter(torch.zeros(input_size))

        # Level 2 gates (expanded features)
        self.gate_expanded = nn.Parameter(torch.zeros(self.expanded_size))

        # Predictor
        layers = []
        in_dim = self.expanded_size
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Mish(), nn.Dropout(0.1)])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.predictor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand
        x_np = x.detach().cpu().numpy()
        x_exp = torch.tensor(
            self.poly.transform(x_np), dtype=x.dtype, device=x.device
        )

        # Two-level gating
        g_orig = torch.sigmoid(self.gate_original)
        g_exp = torch.sigmoid(self.gate_expanded)

        # Map original gates to expanded
        g_full = torch.zeros(self.expanded_size, device=x.device)
        for orig_idx, exp_indices in self.expansion_groups.items():
            g_full[exp_indices] = g_orig[orig_idx] * g_exp[exp_indices]

        return self.predictor(x_exp * g_full)

    def get_selected_features(self) -> Tuple[list[int], list[int]]:
        """Get selected original and expanded feature indices."""
        with torch.no_grad():
            # Level 1: Top-k original
            orig_scores = torch.abs(self.gate_original)
            top_orig = torch.topk(orig_scores, self.k_original).indices.tolist()

            # Level 2: Within each selected original, top-k expanded
            selected_expanded = []
            for orig_idx in top_orig:
                exp_indices = self.expansion_groups[orig_idx]
                exp_scores = torch.abs(self.gate_expanded[exp_indices])
                top_k = min(self.k_expanded_per_feature, len(exp_indices))
                top_exp_local = torch.topk(exp_scores, top_k).indices.tolist()
                selected_expanded.extend([exp_indices[i] for i in top_exp_local])

        return top_orig, selected_expanded


# ---------------------------------------------------------------------------
# Convenience function to run polynomial FS experiment
# ---------------------------------------------------------------------------


def run_polynomial_fs_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    degree: int = 2,
    selection_mode: str = "group",
    C: float = 0.1,
    epochs: int = 200,
    warmup_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> dict:
    """
    Run polynomial feature selection experiment.

    Args:
        degree: Polynomial degree for expansion
        selection_mode: "expanded", "group", or "hierarchical"
        C: Sparsity coefficient

    Returns:
        results: Dict with scores, predictions, etc.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_size = X_train.shape[1]

    # Create model
    model = PolynomialFeatureSelectionModel(
        input_size=input_size,
        n_classes=n_classes,
        degree=degree,
        selection_mode=selection_mode,
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with group lasso
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long if n_classes > 2 else torch.float32, device=device)

    criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_t)

        # Task loss
        if n_classes > 2:
            task_loss = criterion(logits, y_t)
        else:
            task_loss = criterion(logits.squeeze(), y_t)

        # Sparsity loss
        if selection_mode == "group":
            sparsity_loss = C * torch.norm(model.gate, p=1)
        else:
            sparsity_loss = C * torch.norm(model.gate_values(), p=1)

        loss = task_loss + sparsity_loss
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            alive = (torch.abs(model.gate_values()) > 1e-4).sum().item()
            print(f"Epoch {epoch}: loss={loss.item():.4f}, alive={alive}")

    # Get final scores
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_pred = model(X_test_t)
        scores = model.get_original_feature_scores().cpu().numpy()

    return {
        "model": model,
        "scores": scores,
        "predictions": y_pred.cpu().numpy(),
        "expanded_size": model.expanded_size,
        "feature_names": model.feature_names,
    }