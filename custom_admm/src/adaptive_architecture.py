# -*- coding: utf-8 -*-
"""
Adaptive Architecture for Feature Selection.

This module implements dimension-adaptive model architectures that automatically
adjust their capacity based on the input feature dimension.

Key Innovation:
- DimensionAdaptiveGate: Gate network with bottleneck scaled to √(n_features)
- AdaptiveFeatureSelector: Full model combining adaptive gate with fixed backbone

Theoretical Foundation:
- Capacity-Dimension Matching: d_h = ⌈√(m · k)⌉
- Information Bottleneck: Limited capacity forces feature competition

References:
- LassoNet: Lemhadri et al., 2021 (uses 1-layer, 32 units)
- Information Bottleneck: Tishby & Zaslavsky, 2015
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .nn_wrapper import GaussianNoise, init_weights


class DimensionAdaptiveGate(nn.Module):
    """
    Gate network with adaptive bottleneck dimension.

    The gate hidden dimension is scaled as:
        d_g = max(16, ⌈√(n_features)⌉)

    This ensures:
    - Low-dimensional tasks: Simple gate (d_g ≈ 4-16), prevents overfitting
    - High-dimensional tasks: Larger gate (d_g ≈ 24-32), captures complex interactions

    Args:
        n_features: Number of input features
        n_classes: Number of output classes (default: 2)
        gate_bottleneck_ratio: Alternative scaling factor (default: None, uses √n)
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        gate_bottleneck_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        # Compute adaptive gate hidden dimension
        if gate_bottleneck_ratio is not None:
            # Use ratio-based scaling
            self.gate_hidden = max(16, int(n_features * gate_bottleneck_ratio))
        else:
            # Use square root scaling (default)
            self.gate_hidden = max(16, int(np.sqrt(n_features)))

        # Gate network with bottleneck
        # Input → gate_hidden → n_features
        self.gate_net = nn.Sequential(
            nn.Linear(n_features, self.gate_hidden),
            nn.ReLU(),
            nn.Linear(self.gate_hidden, n_features),
        )

        # Initialize gate weights
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                init_weights(module)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive gate and apply to input.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            gate: Gate values after sigmoid, shape (batch_size, n_features)
            x_gated: Gated input, shape (batch_size, n_features)
        """
        gate = torch.sigmoid(self.gate_net(x))
        x_gated = x * gate
        return gate, x_gated


class AdaptiveFeatureSelector(nn.Module):
    """
    Complete adaptive feature selection model.

    Architecture:
        Input → Adaptive Gate (√n bottleneck) → Fixed Backbone (64→32→output)

    Key Features:
    - Dimension-adaptive gate capacity
    - Fixed backbone for fair comparison across dimensions
    - Feature dropout during training
    - Optional Gaussian noise for regularization
    - Compatible with ADMM training

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        backbone_dims: Hidden dimensions for backbone (default: [64, 32])
        activation: Activation function ('relu', 'mish', 'tanh')
        dropout: Dropout probability (default: 0.0)
        gaussian_noise: Input noise std (default: 0.0)
        feat_drop: Feature dropout probability (default: 0.6)
        bounded_gate: If True, gate is sigmoid-bounded to [0,1]
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        backbone_dims: list = None,
        activation: str = "relu",
        dropout: float = 0.0,
        gaussian_noise: float = 0.0,
        feat_drop: float = 0.6,
        bounded_gate: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.backbone_dims = backbone_dims or [64, 32]
        self.activation_name = activation
        self.dropout = dropout
        self.gaussian_noise = gaussian_noise
        self.feat_drop = feat_drop
        self.bounded_gate = bounded_gate

        # Input noise (training only)
        if gaussian_noise > 0:
            self.input_noise = GaussianNoise(gaussian_noise)
        else:
            self.input_noise = None

        # Adaptive gate
        self.gate = DimensionAdaptiveGate(n_features, n_classes)

        # Activation function
        activation_map = {
            "relu": nn.ReLU,
            "mish": nn.Mish,
            "tanh": nn.Tanh,
            "leakyrelu": lambda: nn.LeakyReLU(0.01),
            "selu": nn.SELU,
        }
        activation_fn = activation_map.get(activation, nn.ReLU)

        # Fixed backbone
        layers = []
        prev_dim = n_features
        for hidden_dim in self.backbone_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    activation_fn(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (1 neuron for binary classification)
        n_out = 1 if n_classes <= 2 else n_classes
        layers.append(nn.Linear(prev_dim, n_out))

        self.backbone = nn.Sequential(*layers)

        # Initialize backbone weights
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                init_weights(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional feature dropout.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            Output logits of shape (batch_size, n_classes)
        """
        # Apply input noise (training only)
        if self.input_noise is not None and self.training:
            x = self.input_noise(x)

        # Apply adaptive gate
        gate, x = self.gate(x)

        # Apply feature dropout (training only)
        if self.training and self.feat_drop > 0:
            mask = (torch.rand_like(gate) > self.feat_drop).float()
            x = x * mask / (1.0 - self.feat_drop + 1e-8)

        # Pass through backbone
        out = self.backbone(x)
        return out

    def get_gate_values(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get per-feature importance scores.

        For the adaptive gate, we compute the average gate activation.
        If ``x`` is provided, importance is estimated on real samples;
        otherwise we fall back to random probe inputs.

        Returns:
            Feature importance scores of shape (n_features,)
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            if x is None:
                x_probe = torch.randn(100, self.n_features, device=device)
            else:
                x_probe = x.to(device)
            gate, _ = self.gate(x_probe)
            avg_gate = gate.mean(dim=0).abs()
        if was_training:
            self.train()
        return avg_gate

    def get_gate_params(self) -> torch.Tensor:
        """
        Get raw gate parameters (for compatibility with GatedFeatureSelectionMLP).

        Since our gate is a network, we approximate by returning the first
        layer weights which determine feature importance.

        Returns:
            Gate parameters of shape (n_features,)
        """
        # Use the first layer's input-to-hidden weights
        first_layer = self.gate.gate_net[0]
        # Sum of absolute weights for each input feature
        importance = first_layer.weight.abs().sum(dim=0)
        return importance


class AdaptiveFeatureSelectionMLP(nn.Module):
    """
    Alternative adaptive model with learnable gate parameter (like GatedFeatureSelectionMLP).

    This model combines:
    - Learnable per-feature gate parameter (initialized to 1.0)
    - Adaptive MLP capacity based on input dimension

    Architecture:
        gate (n_features) × Input → Feature Dropout →
        Adaptive MLP (hidden_dim = max(32, √(n_features * k))) → Output

    Args:
        input_size: Number of input features
        n_classes: Number of output classes
        n_hidden_layers: Number of hidden layers (default: adaptive)
        latent_size: Hidden layer dimension (default: adaptive)
        activation: Activation function
        dropout: Dropout probability
        gaussian_noise: Input noise std
        feat_drop: Feature dropout probability
        bounded_gate: If True, apply sigmoid to gate
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int = 2,
        n_hidden_layers: Optional[int] = None,
        latent_size: Optional[int] = None,
        activation: str = "mish",
        dropout: float = 0.043,
        gaussian_noise: float = 0.0,
        feat_drop: float = 0.6,
        bounded_gate: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # Adaptive capacity if not specified
        if latent_size is None:
            # Capacity-Dimension Matching: d_h = ⌈√(m · k)⌉
            # Assume k ≈ 4 (average number of true features)
            self.latent_size = max(32, int(np.sqrt(input_size * 4)))
        else:
            self.latent_size = latent_size

        if n_hidden_layers is None:
            # Scale depth with dimension: 2-5 layers based on input size
            if input_size < 32:
                self.n_hidden_layers = 2
            elif input_size < 256:
                self.n_hidden_layers = 3
            else:
                self.n_hidden_layers = 5
        else:
            self.n_hidden_layers = n_hidden_layers

        self.activation_name = activation
        self.dropout = dropout
        self.gaussian_noise = gaussian_noise
        self.feat_drop = feat_drop
        self.bounded_gate = bounded_gate

        # Learnable gate parameter
        self.gate = nn.Parameter(torch.ones(input_size))

        # Input noise
        if gaussian_noise > 0:
            self.input_noise = GaussianNoise(gaussian_noise)
        else:
            self.input_noise = None

        # Activation function
        activation_map = {
            "relu": nn.ReLU,
            "mish": nn.Mish,
            "tanh": nn.Tanh,
            "leakyrelu": lambda: nn.LeakyReLU(0.01),
            "selu": nn.SELU,
            "gelu": nn.GELU,
        }
        activation_fn = activation_map.get(activation, nn.Mish)

        # Build hidden layers
        layers = []
        prev_dim = input_size
        for _ in range(self.n_hidden_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, self.latent_size),
                    activation_fn(),
                    nn.LayerNorm(self.latent_size),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = self.latent_size

        # Output layer (1 neuron for binary classification)
        n_out = 1 if n_classes <= 2 else n_classes
        layers.append(nn.Linear(prev_dim, n_out))

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        for module in self.layers:
            if isinstance(module, nn.Linear):
                init_weights(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gating and feature dropout."""
        # Apply input noise (training only)
        if self.input_noise is not None and self.training:
            x = self.input_noise(x)

        # Apply gate
        g = torch.sigmoid(self.gate) if self.bounded_gate else self.gate

        # Apply feature dropout (training only)
        if self.training and self.feat_drop > 0:
            mask = (torch.rand_like(g) > self.feat_drop).float()
            g = g * mask / (1.0 - self.feat_drop + 1e-8)

        # Apply gate to input
        x = x * g

        # Pass through layers
        out = self.layers(x)
        return out

    def get_gate_values(self) -> torch.Tensor:
        """Get gate values as feature importance."""
        if self.bounded_gate:
            return torch.sigmoid(self.gate).abs()
        return self.gate.abs()

    def get_gate_params(self) -> torch.Tensor:
        """Get raw gate parameters."""
        return self.gate


# ---------------------------------------------------------------------------
# Model factory function
# ---------------------------------------------------------------------------


def create_adaptive_model(
    n_features: int, n_classes: int = 2, model_type: str = "adaptive_gate", **kwargs
) -> nn.Module:
    """
    Factory function for creating adaptive models.

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        model_type: Type of model ('adaptive_gate', 'adaptive_mlp')
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: Created model

    Examples:
        # Adaptive gate with fixed backbone
        model = create_adaptive_model(n_features=128, model_type="adaptive_gate")

        # Adaptive MLP with capacity scaling
        model = create_adaptive_model(n_features=128, model_type="adaptive_mlp")
    """
    if model_type == "adaptive_gate":
        return AdaptiveFeatureSelector(n_features, n_classes, **kwargs)
    elif model_type == "adaptive_mlp":
        return AdaptiveFeatureSelectionMLP(n_features, n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Utility functions for experiment scripts
# ---------------------------------------------------------------------------


def get_adaptive_capacity(n_features: int, k_estimate: int = 4) -> int:
    """
    Compute adaptive hidden dimension using Capacity-Dimension Matching.

    Formula: d_h = ⌈√(m · k)⌉

    Args:
        n_features: Number of input features (m)
        k_estimate: Estimated number of true features (k)

    Returns:
        Recommended hidden dimension

    Examples:
        >>> get_adaptive_capacity(8)
        32  # min bound
        >>> get_adaptive_capacity(128)
        45
        >>> get_adaptive_capacity(1024)
        64
    """
    return max(32, int(np.ceil(np.sqrt(n_features * k_estimate))))


def get_adaptive_depth(n_features: int) -> int:
    """
    Compute adaptive number of hidden layers.

    Heuristic:
    - m < 32: 2 layers
    - 32 ≤ m < 256: 3 layers
    - m ≥ 256: 5 layers

    Args:
        n_features: Number of input features

    Returns:
        Recommended number of hidden layers
    """
    if n_features < 32:
        return 2
    elif n_features < 256:
        return 3
    else:
        return 5


def get_medium_model_config(n_features: int) -> dict:
    """
    Get configuration for medium-sized model (3-layer, 48 units).

    Args:
        n_features: Number of input features

    Returns:
        Dictionary of model configuration
    """
    return {
        "n_hidden_layers": 3,
        "latent_size": 48,
        "gaussian_noise": 0.0,
        "dropout": 0.043,
        "activation": "mish",
    }
