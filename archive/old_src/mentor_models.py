"""Mentor-driven experimental backbones for gated feature selection.

These models keep a global per-feature gate so they remain compatible with the
existing ADMM training path, while changing either the backbone or the
feature-processing order.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    raise ValueError(f"Unsupported activation: {name}")


class _BaseGatedModel(nn.Module):
    def __init__(
        self, input_size: int, feat_drop: float = 0.6, bounded_gate: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.feat_drop = feat_drop
        self.bounded_gate = bounded_gate
        self.gate = nn.Parameter(torch.ones(input_size))

    def gate_from_parameter(self, gate_param: torch.Tensor) -> torch.Tensor:
        """Convert raw gate parameter to effective gate value."""
        if self.bounded_gate:
            return torch.sigmoid(gate_param)
        return gate_param

    def parameter_from_gate(self, gate_value: torch.Tensor) -> torch.Tensor:
        """Convert effective gate value back to raw parameter."""
        if self.bounded_gate:
            gate_value = torch.clamp(gate_value, min=1e-6, max=1.0 - 1e-6)
            return torch.logit(gate_value)
        return gate_value

    def gate_values(self, training_drop: bool = True) -> torch.Tensor:
        g = torch.sigmoid(self.gate) if self.bounded_gate else self.gate
        if self.training and training_drop and self.feat_drop > 0:
            mask = (torch.rand_like(g) > self.feat_drop).float()
            g = g * mask / (1.0 - self.feat_drop + 1e-8)
        return g


class ExpandedFeatureSelectionMLP(_BaseGatedModel):
    """Feature expansion -> feature selection -> MLP.

    IMPORTANT: The expansion must include a nonlinearity to be meaningful.
    Without it, x @ W_expand @ W_next collapses algebraically to x @ W_combined,
    making this just a reparameterized baseline, not a true expansion test.
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int = 2,
        *,
        expand_dim: int = 8,
        latent_size: int = 64,
        n_hidden_layers: int = 2,
        feat_drop: float = 0.6,
        bounded_gate: bool = False,
        activation: str = "mish",
        dropout: float = 0.0,
    ):
        super().__init__(
            input_size=input_size, feat_drop=feat_drop, bounded_gate=bounded_gate
        )
        self.expand_dim = expand_dim
        self.n_classes = n_classes

        self.expansion_weight = nn.Parameter(torch.empty(input_size, expand_dim))
        nn.init.xavier_uniform_(self.expansion_weight)

        # CRITICAL: Nonlinearity after expansion to prevent algebraic collapse
        self.expansion_activation = _make_activation(activation)

        layers: list[nn.Module] = []
        in_dim = input_size * expand_dim
        act = _make_activation(activation)
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, latent_size))
            layers.append(act.__class__())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = latent_size
        layers.append(nn.Linear(in_dim, 1 if n_classes <= 2 else n_classes))
        self.layers = nn.Sequential(*layers)

    @property
    def first_linear(self) -> nn.Module:
        for mod in self.layers:
            if isinstance(mod, nn.Linear):
                return mod
        raise RuntimeError("No Linear layer found")

    def get_feature_scores(self) -> torch.Tensor:
        return torch.norm(self.expansion_weight, p=2, dim=1) + 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_values(training_drop=True)
        # Expand each feature, then apply nonlinearity
        tokens = x.unsqueeze(-1) * self.expansion_weight.unsqueeze(0)
        tokens = tokens * g.view(1, self.input_size, 1)
        # Apply nonlinearity BEFORE flattening to prevent collapse
        tokens = self.expansion_activation(tokens)
        return self.layers(tokens.reshape(x.shape[0], -1))


class GatedTokenTransformerFS(_BaseGatedModel):
    """Feature-token encoder with a transformer backbone and global ADMM gate."""

    def __init__(
        self,
        input_size: int,
        n_classes: int = 2,
        *,
        d_model: int = 16,
        n_heads: int = 4,
        n_layers: int = 1,
        ff_dim: int = 64,
        feat_drop: float = 0.6,
        bounded_gate: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_size=input_size, feat_drop=feat_drop, bounded_gate=bounded_gate
        )
        self.d_model = d_model
        self.n_classes = n_classes

        self.feature_embedding = nn.Parameter(torch.empty(input_size, d_model))
        nn.init.xavier_uniform_(self.feature_embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, input_size + 1, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1 if n_classes <= 2 else n_classes),
        )
        self.reconstruction_head = nn.Linear(d_model, 1)

    @property
    def first_linear(self) -> nn.Module:
        return self.classifier[-1]

    def get_feature_scores(self) -> torch.Tensor:
        # Tie ADMM penalties to the actual feature selector (gate), not embedding scale.
        # This prevents the model from inflating embedding norms to reduce penalty pressure
        # while keeping feature selection based on gate values.
        return torch.abs(self.gate_from_parameter(self.gate)) + 1e-8

    def _feature_tokens(
        self,
        x: torch.Tensor,
        *,
        apply_gate: bool,
        training_drop: bool,
    ) -> torch.Tensor:
        values = x
        if apply_gate:
            g = self.gate_values(training_drop=training_drop)
            values = values * g.view(1, self.input_size)
        tokens = values.unsqueeze(-1) * self.feature_embedding.unsqueeze(0)
        return tokens

    def encode(
        self,
        x: torch.Tensor,
        *,
        apply_gate: bool = True,
        training_drop: bool = True,
    ) -> torch.Tensor:
        tokens = self._feature_tokens(
            x, apply_gate=apply_gate, training_drop=training_drop
        )
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = seq + self.position_embedding[:, : seq.shape[1], :]
        return self.encoder(seq)

    def encode_feature_tokens(
        self,
        x: torch.Tensor,
        *,
        apply_gate: bool = True,
        training_drop: bool = True,
    ) -> torch.Tensor:
        encoded = self.encode(x, apply_gate=apply_gate, training_drop=training_drop)
        return encoded[:, 1:, :]

    def reconstruct_masked(self, x_masked: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_feature_tokens(
            x_masked,
            apply_gate=False,
            training_drop=False,
        )
        return self.reconstruction_head(encoded).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x, apply_gate=True, training_drop=True)
        cls_state = encoded[:, 0, :]
        return self.classifier(cls_state)
