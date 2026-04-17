# -*- coding: utf-8 -*-
"""
Transformer Pretrain for Feature Selection.

Implements masked feature reconstruction pretraining (MAE-style)
followed by fine-tuning with ADMM gate.

Goal: Determine if transformer backbone can be salvaged with proper pretraining.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Masked Feature Reconstruction Pretraining
# ---------------------------------------------------------------------------


class MaskedFeaturePretrainer(nn.Module):
    """
    Pretrain transformer by reconstructing masked features.

    Similar to MAE but for tabular features:
    1. Randomly mask some features
    2. Encode visible features through transformer
    3. Reconstruct masked features from encoded representation
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 16,
        n_heads: int = 4,
        n_layers: int = 1,
        ff_dim: int = 64,
        dropout: float = 0.1,
        mask_ratio: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # Feature embedding: map each scalar feature to d_model vector
        self.feature_embedding = nn.Parameter(torch.empty(input_size, d_model))
        nn.init.xavier_uniform_(self.feature_embedding)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Position embedding
        self.position_embedding = nn.Parameter(torch.zeros(1, input_size + 1, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)

        # Mask token (learnable placeholder for masked features)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Reconstruction head: map d_model back to scalar
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with masking.

        Args:
            x: (batch, input_size) input features
            mask: (batch, input_size) boolean mask, True = masked

        Returns:
            reconstructed: (batch, input_size) reconstructed features
            loss: scalar reconstruction loss
            mask: the mask used
        """
        batch_size, n_features = x.shape

        # Generate random mask if not provided
        if mask is None:
            mask = torch.rand(batch_size, n_features, device=x.device) < self.mask_ratio

        # Create tokens: embed features
        # x: (batch, n_features) -> tokens: (batch, n_features, d_model)
        tokens = x.unsqueeze(-1) * self.feature_embedding.unsqueeze(0)

        # Replace masked positions with mask token
        mask_token_expanded = self.mask_token.expand(batch_size, n_features, -1)
        tokens = torch.where(
            mask.unsqueeze(-1),
            mask_token_expanded,
            tokens,
        )

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        # Add position embedding
        seq = seq + self.position_embedding[:, : n_features + 1, :]

        # Encode
        encoded = self.encoder(seq)

        # Get feature tokens (exclude CLS)
        feature_tokens = encoded[:, 1:, :]

        # Reconstruct
        reconstructed = self.recon_head(feature_tokens).squeeze(-1)

        # Loss: only on masked positions
        loss = F.mse_loss(
            reconstructed[mask],
            x[mask],
        )

        return reconstructed, loss, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode without masking (for downstream tasks).

        Returns:
            cls_state: (batch, d_model) CLS token representation
        """
        batch_size = x.shape[0]
        tokens = x.unsqueeze(-1) * self.feature_embedding.unsqueeze(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = seq + self.position_embedding[:, : seq.shape[1], :]
        encoded = self.encoder(seq)
        return encoded[:, 0, :]


def pretrain_transformer(
    X_train: np.ndarray,
    d_model: int = 16,
    n_heads: int = 4,
    n_layers: int = 1,
    ff_dim: int = 64,
    mask_ratio: float = 0.3,
    lr: float = 1e-5,
    epochs: int = 100,
    batch_size: int = 64,
    weight_decay: float = 0.01,
    device: Optional[str] = None,
    verbose: bool = True,
) -> MaskedFeaturePretrainer:
    """
    Pretrain transformer with masked feature reconstruction.

    Args:
        X_train: Training data
        lr: Learning rate (start small, e.g., 1e-5)
        mask_ratio: Fraction of features to mask

    Returns:
        model: Pretrained model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_size = X_train.shape[1]

    model = MaskedFeaturePretrainer(
        input_size=input_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        mask_ratio=mask_ratio,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for (x_batch,) in loader:
            reconstructed, loss, mask = model(x_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: recon_loss = {avg_loss:.6f}")

    return model


# ---------------------------------------------------------------------------
# Fine-tuning with ADMM Gate
# ---------------------------------------------------------------------------


class PretrainedTransformerWithGate(nn.Module):
    """
    Pretrained transformer encoder + ADMM gate for feature selection.
    """

    def __init__(
        self,
        pretrained_encoder: MaskedFeaturePretrainer,
        n_classes: int = 2,
        feat_drop: float = 0.6,
        bounded_gate: bool = False,
    ):
        super().__init__()
        self.input_size = pretrained_encoder.input_size
        self.d_model = pretrained_encoder.d_model
        self.n_classes = n_classes
        self.bounded_gate = bounded_gate
        self.feat_drop = feat_drop

        # Copy pretrained components
        self.feature_embedding = nn.Parameter(pretrained_encoder.feature_embedding.data.clone())
        self.cls_token = nn.Parameter(pretrained_encoder.cls_token.data.clone())
        self.position_embedding = nn.Parameter(pretrained_encoder.position_embedding.data.clone())
        self.encoder = pretrained_encoder.encoder

        # ADMM gate (new, not pretrained)
        self.gate = nn.Parameter(torch.zeros(self.input_size))

        # Classifier head (new, not pretrained)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1 if n_classes <= 2 else n_classes),
        )

    def gate_values(self, training_drop: bool = True) -> torch.Tensor:
        """Get effective gate values with optional dropout."""
        g = torch.sigmoid(self.gate) if self.bounded_gate else self.gate

        if training_drop and self.training and self.feat_drop > 0:
            # Feature dropout: randomly drop gates
            mask = torch.rand_like(g) > self.feat_drop
            g = g * mask / (1 - self.feat_drop)

        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Apply gate
        g = self.gate_values(training_drop=True)
        x_gated = x * g

        # Create tokens
        tokens = x_gated.unsqueeze(-1) * self.feature_embedding.unsqueeze(0)

        # Prepend CLS
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = seq + self.position_embedding[:, : seq.shape[1], :]

        # Encode
        encoded = self.encoder(seq)
        cls_state = encoded[:, 0, :]

        # Classify
        return self.classifier(cls_state)

    def get_feature_scores(self) -> torch.Tensor:
        """Return feature importance scores."""
        return torch.abs(self.gate)

    @property
    def first_linear(self) -> nn.Module:
        """For compatibility with ADMM training."""
        return self.classifier[-1]


def finetune_with_admm(
    pretrained_model: MaskedFeaturePretrainer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    C: float = 0.1,
    epochs: int = 200,
    warmup_epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 64,
    rho_init: float = 200.0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> PretrainedTransformerWithGate:
    """
    Fine-tune pretrained transformer with ADMM gate.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create gated model from pretrained
    model = PretrainedTransformerWithGate(pretrained_model, n_classes=n_classes)
    model.to(device)

    # Data
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long if n_classes > 2 else torch.float32, device=device)
    loader = DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss
    criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()

    # ADMM variables
    z = torch.zeros(model.input_size, device=device)
    u = torch.zeros(model.input_size, device=device)
    rho = rho_init

    for epoch in range(epochs):
        model.train()

        for x_batch, y_batch in loader:
            # Forward
            logits = model(x_batch)

            # Task loss
            if n_classes > 2:
                task_loss = criterion(logits, y_batch)
            else:
                task_loss = criterion(logits.squeeze(), y_batch.float())

            # ADMM augmented loss
            g = model.gate_values(training_drop=False)
            admm_loss = (rho / 2) * torch.norm(g - z + u) ** 2

            loss = task_loss + admm_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # z-update: proximal ratio norm
        with torch.no_grad():
            g = model.gate_values(training_drop=False)
            z = _proximal_ratio_norm(g + u, C / rho)

        # Dual update
        u = u + g - z

        if verbose and epoch % 20 == 0:
            alive = (torch.abs(model.gate) > 1e-4).sum().item()
            print(f"Epoch {epoch}: loss={loss.item():.4f}, alive={alive}")

    return model


def _proximal_ratio_norm(v: torch.Tensor, lambda_: float) -> torch.Tensor:
    """
    Proximal operator for ratio norm.

    Approximation: soft thresholding + scaling.
    """
    # Soft threshold
    thresholded = torch.sign(v) * torch.clamp(torch.abs(v) - lambda_, min=0)

    # Scale to unit norm if not zero
    norm = torch.norm(thresholded)
    if norm > 1e-8:
        return thresholded
    return thresholded


# ---------------------------------------------------------------------------
# Full Experiment Runner
# ---------------------------------------------------------------------------


def run_transformer_pretrain_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    d_model: int = 16,
    n_heads: int = 4,
    n_layers: int = 1,
    mask_ratio: float = 0.3,
    pretrain_lr: float = 1e-5,
    pretrain_epochs: int = 100,
    finetune_lr: float = 1e-4,
    finetune_epochs: int = 200,
    C: float = 0.1,
    device: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run full transformer pretrain + finetune experiment.

    Args:
        pretrain_lr: Learning rate for pretraining (try 1e-5)
        finetune_lr: Learning rate for finetuning
        C: Sparsity coefficient

    Returns:
        results: Dict with pretrain_loss, scores, predictions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("=== Stage 1: Pretraining ===")

    # Pretrain
    pretrained = pretrain_transformer(
        X_train,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mask_ratio=mask_ratio,
        lr=pretrain_lr,
        epochs=pretrain_epochs,
        device=device,
        verbose=verbose,
    )

    if verbose:
        print("\n=== Stage 2: Fine-tuning with ADMM ===")

    # Finetune
    model = finetune_with_admm(
        pretrained,
        X_train,
        y_train,
        n_classes,
        C=C,
        epochs=finetune_epochs,
        lr=finetune_lr,
        device=device,
        verbose=verbose,
    )

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits = model(X_test_t)
        scores = model.get_feature_scores().cpu().numpy()

    return {
        "pretrained": pretrained,
        "model": model,
        "scores": scores,
        "logits": logits.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Hyperparameter Search
# ---------------------------------------------------------------------------


def get_transformer_hyperparameter_grid() -> dict:
    """Return suggested hyperparameter grid for sweep."""
    return {
        "pretrain_lr": [1e-6, 1e-5, 5e-5, 1e-4],
        "finetune_lr": [1e-5, 1e-4, 5e-4],
        "mask_ratio": [0.1, 0.3, 0.5, 0.7],
        "d_model": [8, 16, 32],
        "n_layers": [1, 2, 3],
        "n_heads": [2, 4, 8],
        "C": [0.01, 0.1, 1.0],
    }


def get_recommended_config() -> dict:
    """Return recommended starting configuration."""
    return {
        "pretrain_lr": 1e-5,
        "finetune_lr": 1e-4,
        "mask_ratio": 0.3,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 4,
        "C": 0.1,
        "pretrain_epochs": 100,
        "finetune_epochs": 200,
    }