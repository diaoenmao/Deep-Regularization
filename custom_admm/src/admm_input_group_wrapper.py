# -*- coding: utf-8 -*-
"""
Clean ADMM / Lasso feature-selection wrapper for admm_input_group only.

This is a minimal version that only supports the admm_input_group method
with its internal implementation, without dependencies on old optimizers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from local src directory
from src.nn_wrapper import GaussianNoise
from src.utils import TestSet, TrainingSet

# Import ADMM utilities from sibling module
from .admm_utils import (
    safe_cbrt,
    safe_norm,
    soft_thresholding,
    solve_cubic_paper,
    solve_cubic_ratio_norm,
)

# Import adaptive architectures (optional, for comparison experiments)
try:
    from .adaptive_architecture import (
        AdaptiveFeatureSelectionMLP,
        AdaptiveFeatureSelector,
        DimensionAdaptiveGate,
        create_adaptive_model,
    )

    ADAPTIVE_MODELS_AVAILABLE = True
except ImportError:
    ADAPTIVE_MODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# MLP model (mirrors the benchmark's ``Model`` but simplified for FS)
# ---------------------------------------------------------------------------

DEFAULT_SADMM_LATENT_SIZE = 32
DEFAULT_SADMM_HIDDEN_LAYERS = 2
DEFAULT_SADMM_DROPOUT = 0.04308691548552568
DEFAULT_SADMM_ACTIVATION = "mish"


class ColumnNormalizedLinear(nn.Module):
    """Linear layer that uses unit-norm input columns during forward passes."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.raw_weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.raw_weight)
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        col_norm = torch.norm(self.raw_weight, p=2, dim=0, keepdim=True).clamp_min(1e-8)
        return self.raw_weight / col_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class FeatureSelectionMLP(nn.Module):
    """Simple MLP for tabular classification, compatible with sparsity optimizers."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,
        dropout: float = DEFAULT_SADMM_DROPOUT,
        activation: str = DEFAULT_SADMM_ACTIVATION,
    ):
        super().__init__()
        n_out = 1 if n_classes <= 2 else n_classes

        layers: list[nn.Module] = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))

        for k in range(n_hidden_layers):
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = input_size if k == 0 else latent_size
            layers.append(nn.Linear(in_dim, latent_size))

            act: nn.Module
            if activation == "relu":
                act = nn.ReLU()
            elif activation == "leakyrelu":
                act = nn.LeakyReLU(0.2)
            elif activation == "mish":
                act = nn.Mish()
            elif activation == "selu":
                act = nn.SELU()
            else:
                act = nn.Mish()
            layers.append(act)

        layers.append(nn.Linear(latent_size, n_out))
        self.layers = nn.Sequential(*layers)
        self.apply(init_weights)

    # Convenience: find the first Linear layer (for feature importance)
    @property
    def first_linear(self) -> nn.Module:
        for m in self.layers:
            if isinstance(m, (nn.Linear, ColumnNormalizedLinear)):
                return m
        raise RuntimeError("No Linear layer found")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GatedFeatureSelectionMLP(nn.Module):
    """MLP with learnable per-feature gates and input feature dropout.

    During training, each input feature is randomly masked with probability
    ``feat_drop``.  This prevents the model from memorising via irrelevant
    features (critical when n_features >> n_samples).  The learnable gate
    ``theta`` is pruned via Linearized ADMM + Ratio Norm to identify
    informative features.

    The paper's main benchmark uses one fixed compact predictor
    (2 hidden layers, width 32) without per-dataset retuning.
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = DEFAULT_SADMM_LATENT_SIZE,
        n_hidden_layers: int = DEFAULT_SADMM_HIDDEN_LAYERS,
        gaussian_noise: float = 0.0,  # Disabled by default; set >0 to enable
        dropout: float = DEFAULT_SADMM_DROPOUT,
        feat_drop: float = 0.6,  # Tuned value (was 0.7); see tune_feat_drop.py ablation
        activation: str = DEFAULT_SADMM_ACTIVATION,
        bounded_gate: bool = False,
        layer_norm: int = 0,
        column_normalize_first_layer: bool = False,
    ):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(input_size))
        self.feat_drop = feat_drop
        self.bounded_gate = bounded_gate
        self.column_normalize_first_layer = column_normalize_first_layer
        n_out = 1 if n_classes <= 2 else n_classes

        layers: list[nn.Module] = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))

        inplace = False
        for k in range(n_hidden_layers):
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout, inplace=inplace))

            in_dim = input_size if k == 0 else latent_size
            if k == 0 and column_normalize_first_layer:
                layers.append(ColumnNormalizedLinear(in_dim, latent_size))
            else:
                layers.append(nn.Linear(in_dim, latent_size))

            if layer_norm:
                layers.append(nn.LayerNorm(latent_size))

            act: nn.Module
            if activation == "relu":
                act = nn.ReLU(inplace=inplace)
            elif activation == "leakyrelu":
                act = nn.LeakyReLU(0.2, inplace=inplace)
            elif activation == "prelu":
                act = nn.PReLU(latent_size)
            elif activation == "tanh":
                act = nn.Tanh()
            elif activation == "sigmoid":
                act = nn.Sigmoid()
            elif activation == "mish":
                act = nn.Mish(inplace=inplace)
            elif activation == "selu":
                act = nn.SELU(inplace=inplace)
            else:
                act = nn.Hardswish(inplace=inplace)
            layers.append(act)

        layers.append(nn.Linear(latent_size, n_out))
        self.layers = nn.Sequential(*layers)
        # Use PyTorch default init (kaiming_uniform_ a=sqrt(5)) â?empirically
        # better for gate-based feature selection than the custom init_weights.

    @property
    def first_linear(self) -> nn.Module:
        for m in self.layers:
            if isinstance(m, (nn.Linear, ColumnNormalizedLinear)):
                return m
        raise RuntimeError("No Linear layer found")

    def gate_from_parameter(self, gate_param: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            return torch.sigmoid(gate_param)
        return gate_param

    def parameter_from_gate(self, gate_value: torch.Tensor) -> torch.Tensor:
        if self.bounded_gate:
            gate_value = torch.clamp(gate_value, min=1e-6, max=1.0 - 1e-6)
            return torch.logit(gate_value)
        return gate_value

    def get_gate_values(self, training_drop: bool = False) -> torch.Tensor:
        g = self.gate_from_parameter(self.gate)
        if self.training and training_drop and self.feat_drop > 0:
            mask = (torch.rand(g.shape, device=g.device) > self.feat_drop).float()
            g = g * mask / (1.0 - self.feat_drop + 1e-8)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x * self.get_gate_values(training_drop=True))


# ---------------------------------------------------------------------------
# Strategy 2: z-score data standardisation
# ---------------------------------------------------------------------------


class _Scaler:
    """Z-score scaler that remembers train statistics."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return (X - self.mean_) / self.std_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


# ---------------------------------------------------------------------------
# WANDA-like score computation for tabular MLP
# ---------------------------------------------------------------------------


def compute_mlp_wanda_scores(
    model: nn.Module, X_sample: torch.Tensor
) -> list[torch.Tensor]:
    """Compute per-parameter WANDA-inspired importance scores.

    For each Linear layer:  score = |W| * ||activation_input||_2
    For other param types:    score = ones (no pruning importance info).

    We run a forward pass on ``X_sample`` to capture activations, then
    combine them with weight magnitudes.

    Returns a list of tensors aligned with ``list(model.parameters())``.
    """
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            activations[name] = inp[0].detach()

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(X_sample)

    for h in hooks:
        h.remove()

    # Build scores list aligned with model.parameters()
    param_to_module: dict[int, str] = {}
    for name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            param_to_module[id(p)] = name

    scores: list[torch.Tensor] = []
    for p in model.parameters():
        mod_name = param_to_module.get(id(p))
        if mod_name and mod_name in activations:
            act = activations[mod_name]  # (batch, in_features)
            act_norm = torch.norm(act, p=2, dim=0)  # (in_features,)
            if p.dim() == 2:
                # Weight matrix: score_ij = |W_ij| * ||a_j||_2
                scores.append(torch.abs(p.data) * act_norm.unsqueeze(0))
            elif p.dim() == 1 and p.shape[0] == act.shape[1]:
                # Bias: just use magnitude
                scores.append(torch.abs(p.data) + 1e-8)
            else:
                scores.append(torch.ones_like(p.data))
        else:
            scores.append(torch.ones_like(p.data))

    return scores


# ---------------------------------------------------------------------------
# Feature score helper for ADMM thresholds
# ---------------------------------------------------------------------------


def _get_feature_penalty_scores(model: nn.Module) -> torch.Tensor:
    """Return one positive score per original input feature.

    Standard MLP variants use the first dense layer column norms.
    Custom backbones can expose `get_feature_scores()` when there is no
    feature-aligned dense first layer.
    """
    device = next(model.parameters()).device
    if hasattr(model, "get_feature_scores"):
        score = model.get_feature_scores()
        if not torch.is_tensor(score):
            score = torch.as_tensor(score, dtype=torch.float32, device=device)
        else:
            score = score.to(device=device, dtype=torch.float32)
        return torch.clamp(score.detach(), min=1e-8)

    W1 = model.first_linear.weight
    return torch.norm(W1, p=2, dim=0) + 1e-8


# ---------------------------------------------------------------------------
# Soft thresholding utility (needed for ADMM)
# ---------------------------------------------------------------------------


def soft_thresholding(v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Soft thresholding operator: sign(v) * max(|v| - alpha, 0)."""
    return torch.sign(v) * torch.clamp(torch.abs(v) - alpha, min=0.0)


def safe_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe L2 norm with epsilon for numerical stability."""
    return torch.norm(v, p=2) + eps


def safe_cbrt(x: torch.Tensor) -> torch.Tensor:
    """Safe cube root that handles negative numbers."""
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


def solve_cubic_ratio_norm(D: torch.Tensor) -> torch.Tensor:
    """Solve ÏÂ³ - Ï - D = 0 for Ï â?0 using Cardano's formula.

    For the equation ÏÂ³ - Ï - D = 0:
    - Discriminant: Î = DÂ²/4 - 1/27
    - If Î >= 0: one real root Ï = â?D/2 + âÎ? + â?D/2 - âÎ?
    - If Î < 0: three real roots, use trigonometric method

    Args:
        D: Input tensor (should be non-negative for Ratio Norm)

    Returns:
        Ï: Positive real root of the cubic equation
    """
    tau = torch.zeros_like(D)

    # Discriminant for ÏÂ³ - Ï - D = 0: Î = DÂ²/4 - 1/27
    discriminant = (D / 2.0) ** 2 - 1.0 / 27.0

    # Case 1: discriminant >= 0 (one real root)
    case1 = discriminant >= 0
    if torch.any(case1):
        sqrt_disc = torch.sqrt(discriminant[case1])
        # Ï = â?D/2 + âÎ? + â?D/2 - âÎ?
        # FIX: Use safe_cbrt to handle negative values
        term1 = D[case1] / 2.0 + sqrt_disc
        term2 = D[case1] / 2.0 - sqrt_disc  # FIX: was -D/2 + sqrt_disc (wrong sign)
        tau[case1] = safe_cbrt(term1) + safe_cbrt(term2)

    # Case 2: discriminant < 0 (three real roots, use trigonometric method)
    case2 = ~case1
    if torch.any(case2):
        # For ÏÂ³ - Ï - D = 0, the trigonometric solution is:
        # Ï = 2/â? * cos(Î¸/3) where Î¸ = arccos(3â? D / 2)
        sqrt_3 = torch.sqrt(torch.tensor(3.0, device=D.device))
        arg = torch.clamp(3.0 * sqrt_3 * D[case2] / 2.0, min=-1.0, max=1.0)
        theta = torch.acos(arg)
        tau[case2] = 2.0 * torch.cos(theta / 3.0) / sqrt_3

    return tau


def solve_cubic_paper(D: torch.Tensor) -> torch.Tensor:
    """Solve cubic equation from the paper formulation.

    This uses the same Cardano's formula as solve_cubic_ratio_norm,
    adapted for the paper's specific cubic equation format.
    """
    return solve_cubic_ratio_norm(D)


# ---------------------------------------------------------------------------
# ADMM-Gate training loop for Input-Group feature selection
# ---------------------------------------------------------------------------


def _train_input_group(
    model,  # GatedFeatureSelectionMLP
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    lr: float = 0.005,
    C: float = 0.1,  # Ratio Norm sparsity coefficient
    epochs: int = 500,
    warmup_epochs: int = 120,
    batch_size: int = 64,
    rho_init: float = 200.0,
    rho_update_interval: int = 5,
    score_refresh_interval: int = 10,
    device: Optional[str] = None,
    use_ratio_norm: bool = True,  # False â?plain L1 proximal (ablation)
    use_admm: bool = True,  # False â?proximal gradient (ablation)
    uniform_penalty: bool = False,  # True â?Î»_j = C (uniform); False â?Î»_j = C/s_j (adaptive)
    three_variable: bool = False,  # True â?paper's 3-variable ADMM (y,z with duals v,w)
    n_features: int = None,  # Feature dimension for rho scaling
    # Phase 1 experimental improvements:
    optimizer_type: str = "adam",  # "adam" or "adagrad"
    use_early_stopping: bool = False,
    patience: int = 66,
    val_split: float = 0.2,  # Validation split for early stopping
) -> None:
    """Train a GatedFeatureSelectionMLP with Linearized ADMM + Ratio Norm.

    **Phase 1 â?Warm-up** (epoch 0 â?warmup_epochs-1):
        All parameters trained with Adam.  Feature dropout (in the model)
        prevents memorisation, letting the MLP learn the true signal.

    **Phase 2 â?Linearized ADMM pruning** (epoch warmup_epochs â?epochs-1):
        Uses 2-variable ADMM splitting:  min L(Î¸,g) + Î»Â·R(z)  s.t. g = z
        where R(z) = âzââ/âzââ is the Ratio Norm (scale-invariant).

        Per epoch:
          1. **g-step**: Adam on ALL params (Î¸ AND g) with augmented loss
             L(Î¸,g) + (Ï/2)âg â?z + uâÂ?  â?gate has full gradient dynamics
          2. **z-step**: z = prox_{Î»/Ï Â· RatioNorm}(g + u)
             â?cubic ÏÂ³ â?Ï â?D = 0 for scaling + soft-thresh for sparsity
          3. **Dual update**: u â?u + g â?z
          4. **Adaptive Ï**: Boyd Â§3.4.1 with proper dual rescaling

        Score s_j = ||W1[:,j]||_2 gives importance-adaptive thresholds
        lambda_j = C/s_j, so the Ratio Norm amplifies signal features and
        suppresses noise -- an effect absent in plain L1.

    **Phase 1 Experimental Improvements:**
        - optimizer_type: "adam" (default) or "adagrad" (benchmark default)
        - use_early_stopping: enable validation-based early stopping
        - patience: number of epochs without improvement before stopping
    """
    N = len(X_train)
    n_features = X_train.shape[1]  # FIX: define n_features for rho_min calculation

    # Validation split for early stopping
    if use_early_stopping and val_split > 0:
        n_val = int(len(X_train) * val_split)
        idx = np.random.permutation(len(X_train))
        X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
        X_train, y_train = X_train[idx[n_val:]], y_train[idx[n_val:]]
    else:
        X_val, y_val = None, None

    dataset = TrainingSet(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if X_val is not None:
        val_dataset = TrainingSet(X_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        val_loader = None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if n_classes <= 2:
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        criterion = nn.NLLLoss(reduction="mean")

    # ==================================================================
    # Phase 1: Warm-up â?Optimizer selection (Adam/Adagrad)
    # ==================================================================
    if optimizer_type == "adam":
        opt_warmup = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    elif optimizer_type == "adagrad":
        opt_warmup = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-3)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    model.train()
    for epoch in range(warmup_epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            opt_warmup.zero_grad()
            y_hat = model(x_batch)
            if n_classes > 2:
                y_hat = torch.log_softmax(y_hat, dim=1)
            else:
                y_hat = y_hat.reshape(len(y_hat))

            try:
                loss = criterion(y_hat, y_batch)
            except RuntimeError:
                loss = criterion(y_hat, y_batch.float())

            loss.backward()
            opt_warmup.step()

    # ==================================================================
    # Phase 2: Linearized ADMM â?gate in Adam + Ratio Norm z-step
    #
    #   min_{g,Î¸}  L(Î¸, g)  +  Î» Â· R(z)           (R = Ratio Norm)
    #   s.t.  g = z
    #
    #   Augmented Lagrangian (scaled form):
    #     L_Ï = L(Î¸, g) + (Ï/2) âg â?z + uâÂ?    #
    #   Per epoch:
    #     (1) g-step:  Adam on (Î¸, g) with augmented loss
    #                  â?gate gets full gradient dynamics
    #     (2) z-step:  z = prox_{Î»/Ï Â· RatioNorm}(g + u)
    #                  â?Ratio Norm proximal (cubic solver) for sparsity
    #     (3) dual:    u â?u + g â?z
    #
    #   Score s_j = âWâ[:,j]ââ gives importance-adaptive thresholds.
    # ==================================================================
    gate_param = model.gate  # shape (m,)
    m = gate_param.shape[0]
    bounded_gate = getattr(model, "bounded_gate", False)

    def _effective_gate(param: torch.Tensor) -> torch.Tensor:
        """Convert raw parameter to effective gate value (apply sigmoid if bounded)."""
        if hasattr(model, "gate_from_parameter"):
            return model.gate_from_parameter(param)
        return param

    def _parameter_from_effective(gate_value: torch.Tensor) -> torch.Tensor:
        """Convert effective gate value to raw parameter (inverse sigmoid if bounded)."""
        if hasattr(model, "parameter_from_gate"):
            return model.parameter_from_gate(gate_value)
        return gate_value

    def _project_effective_gate(gate_value: torch.Tensor) -> torch.Tensor:
        """Project effective gate to valid range. Only used for effective-space ADMM."""
        if bounded_gate:
            return torch.clamp(gate_value, min=0.0, max=1.0)
        return gate_value

    # ADMM buffers in RAW SPACE for bounded_gate, EFFECTIVE SPACE otherwise
    # This is the key fix: bounded_gate ADMM operates in raw space
    if bounded_gate:
        # Raw space ADMM: zk and uk are in raw (pre-sigmoid) space
        zk = gate_param.data.clone()  # raw gate value
    else:
        # Effective space ADMM: zk and uk are in effective space
        zk = _effective_gate(gate_param.data).clone()
    uk = torch.zeros(m, device=device)  # scaled dual variable
    rho = rho_init

    # Score = first-layer column norms (importance per feature)
    with torch.no_grad():
        score = _get_feature_penalty_scores(model)

    # Adam on ALL parameters including gate (gate gets gradient dynamics)
    opt_all = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    prune_epochs = epochs - warmup_epochs

    # ââ Ablation: proximal gradient (no ADMM) ââââââââââââââââââââââââ
    if not use_admm:
        opt_prox = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        for epoch in range(prune_epochs):
            with torch.no_grad():
                score = _get_feature_penalty_scores(model)

            # Forward pass (Adam step)
            model.train()
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                opt_prox.zero_grad()
                y_hat = model(x_batch)
                if n_classes > 2:
                    y_hat = torch.log_softmax(y_hat, dim=1)
                else:
                    y_hat = y_hat.reshape(len(y_hat))
                try:
                    loss = criterion(y_hat, y_batch)
                except RuntimeError:
                    loss = criterion(y_hat, y_batch.float())
                loss.backward()
                opt_prox.step()

            # Proximal step: apply proximal operator directly to gate
            with torch.no_grad():
                # For bounded_gate: use raw gate; for unbounded: use effective gate
                if bounded_gate:
                    g = gate_param.data  # raw space
                else:
                    g = _effective_gate(gate_param.data)
                if uniform_penalty:
                    lam = torch.full_like(score, C)
                else:
                    lam = C / score  # per-feature lambda
                lam = torch.clamp(lam, min=1e-8, max=0.5)
                alpha = lam * lr  # proximal step size
                g_shrunk = soft_thresholding(g, alpha)
                if use_ratio_norm and torch.norm(g_shrunk, p=1) > 1e-8:
                    v_l2 = safe_norm(g_shrunk)
                    mu = C * torch.norm(g_shrunk, p=1) / N
                    D_k = (mu * score * score) / (1.0 * torch.clamp(v_l2**3, min=1e-10))
                    tau_k = solve_cubic_ratio_norm(D_k)
                    g_shrunk = tau_k * g_shrunk
                # For bounded_gate (raw space): no projection needed
                # For unbounded_gate (effective space): project to valid range
                if not bounded_gate:
                    g_shrunk = _project_effective_gate(g_shrunk)
                if bounded_gate:
                    gate_param.data.copy_(g_shrunk)
                else:
                    gate_param.data.copy_(_parameter_from_effective(g_shrunk))
        model.eval()
        return

    # ââ 3-variable ADMM path (paper's formulation) âââââââââââââââââââââ
    if three_variable and use_ratio_norm:
        # For bounded_gate: use raw space; for unbounded: use effective space
        if bounded_gate:
            yk = gate_param.data.clone()
            zk_3 = gate_param.data.clone()
        else:
            yk = _effective_gate(gate_param.data).clone()
            zk_3 = _effective_gate(gate_param.data).clone()
        vk = torch.zeros(m, device=device)  # dual for g=y
        wk = torch.zeros(m, device=device)  # dual for g=z
        convergence_log = []

        for epoch in range(prune_epochs):
            with torch.no_grad():
                score = _get_feature_penalty_scores(model)

            # (1) g-step: Adam on augmented loss with TWO penalty terms
            model.train()
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                opt_all.zero_grad()
                y_hat = model(x_batch)
                if n_classes > 2:
                    y_hat = torch.log_softmax(y_hat, dim=1)
                else:
                    y_hat = y_hat.reshape(len(y_hat))
                try:
                    data_loss = criterion(y_hat, y_batch)
                except RuntimeError:
                    data_loss = criterion(y_hat, y_batch.float())

                # For bounded_gate: use raw gate; for unbounded: use effective gate
                if bounded_gate:
                    gate_value_for_penalty = gate_param
                else:
                    gate_value_for_penalty = _effective_gate(gate_param)
                penalty = (rho / 2.0) * (
                    torch.sum((gate_value_for_penalty - yk.detach() + vk.detach()) ** 2)
                    + torch.sum((gate_value_for_penalty - zk_3.detach() + wk.detach()) ** 2)
                )
                (data_loss + penalty).backward()
                opt_all.step()

            # (2) y-step: cubic update in effective gate space
            with torch.no_grad():
                if uniform_penalty:
                    lam = torch.full_like(score, C)
                else:
                    lam = C / score
                lam = torch.clamp(lam, min=1e-6, max=0.5)

                c_t = torch.sum(lam * torch.abs(zk_3))
                if bounded_gate:
                    d_t = gate_param.data + vk / rho
                else:
                    d_t = _effective_gate(gate_param.data) + vk / rho
                d_norm = safe_norm(d_t)

                if d_norm > 1e-8 and c_t > 1e-8:
                    D_scalar = c_t / (N * rho * torch.clamp(d_norm**3, min=1e-10))
                    tau_y = solve_cubic_paper(D_scalar.unsqueeze(0)).squeeze(0)
                    yk = tau_y * d_t
                else:
                    yk = d_t.clone()
                # For bounded_gate (raw space): no projection needed
                if not bounded_gate:
                    yk = _project_effective_gate(yk)

            # (3) z-step: soft-threshold in effective gate space
            with torch.no_grad():
                zk_3_old = zk_3.clone()
                y_norm = safe_norm(yk)
                if bounded_gate:
                    a = gate_param.data + wk / rho
                else:
                    a = _effective_gate(gate_param.data) + wk / rho
                kappa = lam / (N * rho * torch.clamp(y_norm, min=1e-8))
                zk_3 = soft_thresholding(a, kappa)
                # For bounded_gate (raw space): no projection needed
                if not bounded_gate:
                    zk_3 = _project_effective_gate(zk_3)

            # (4) dual updates
            with torch.no_grad():
                if bounded_gate:
                    gate_value = gate_param.data
                else:
                    gate_value = _effective_gate(gate_param.data)
                vk = vk + rho * (gate_value - yk)
                wk = wk + rho * (gate_value - zk_3)

            with torch.no_grad():
                if bounded_gate:
                    gate_value = gate_param.data
                else:
                    gate_value = _effective_gate(gate_param.data)
                primal_resid = torch.norm(gate_value - zk_3).item()
                dual_resid = rho * torch.norm(zk_3 - zk_3_old).item()
                convergence_log.append((epoch, primal_resid, dual_resid))

            if epoch > 0 and epoch % rho_update_interval == 0:
                with torch.no_grad():
                    gate_value = _effective_gate(gate_param.data)
                    r_norm = max(torch.norm(gate_value - zk_3).item(), 1e-12)
                    s_norm = max(rho * torch.norm(zk_3 - zk_3_old).item(), 1e-12)
                    rho_old = rho
                    if r_norm > 10.0 * s_norm:
                        rho = min(rho * 2.0, 1e4)
                    elif s_norm > 10.0 * r_norm:
                        rho = max(rho / 2.0, 50.0)
                    if rho != rho_old:
                        vk = vk * (rho_old / rho)
                        wk = wk * (rho_old / rho)

        model.eval()
        model.convergence_log = convergence_log
        return

    convergence_log = []  # stores (epoch, primal_resid, dual_resid) for diagnostics

    # Early stopping variables
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_count = 0

    # ADMM optimizer selection
    if optimizer_type == "adam":
        opt_all = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    elif optimizer_type == "adagrad":
        opt_all = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-3)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    for epoch in range(prune_epochs):
        # Refresh score every epoch (column norms evolve with training)
        with torch.no_grad():
            score = _get_feature_penalty_scores(model)

        # ââ (1) g-step: Adam on augmented loss over all mini-batches ââ
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            opt_all.zero_grad()
            y_hat = model(x_batch)
            if n_classes > 2:
                y_hat = torch.log_softmax(y_hat, dim=1)
            else:
                y_hat = y_hat.reshape(len(y_hat))

            try:
                data_loss = criterion(y_hat, y_batch)
            except RuntimeError:
                data_loss = criterion(y_hat, y_batch.float())

            # ADMM augmented Lagrangian penalty on gate
            # For bounded_gate: ADMM operates in RAW SPACE (pre-sigmoid)
            # For unbounded_gate: ADMM operates in EFFECTIVE SPACE
            if bounded_gate:
                # Raw space ADMM: penalty on raw gate parameter
                admm_penalty = (rho / 2.0) * torch.sum(
                    (gate_param - zk.detach() + uk.detach()) ** 2
                )
            else:
                # Effective space ADMM: penalty on effective gate value
                gate_value = _effective_gate(gate_param)
                admm_penalty = (rho / 2.0) * torch.sum(
                    (gate_value - zk.detach() + uk.detach()) ** 2
                )
            total_loss = data_loss + admm_penalty

            total_loss.backward()
            opt_all.step()

        # ââ (2) z-step: Ratio Norm proximal (once per epoch) ââââââââââ
        with torch.no_grad():
            zk_old = zk.clone()
            # For bounded_gate: use raw gate; for unbounded: use effective gate
            if bounded_gate:
                v = gate_param.data + uk  # raw space
            else:
                gate_value = _effective_gate(gate_param.data)
                v = gate_value + uk  # effective space

            # --- Ratio Norm proximal on v ---
            # Ratio Norm R(z) = ||z||â?/ ||z||â? (scale-invariant)
            # prox_{Î»/Ï Â· R}(v) decomposes into:
            #   direction:  solve  ÏÂ³ â?Ï â?D = 0  for scaling
            #   shrinkage:  soft-threshold for L1 component
            #
            # Importance-adaptive Î»_j = C / s_j  (strong on noise,
            # weak on signal â?threshold ~0.05 scale, matching proximal L1)
            if uniform_penalty:
                lam = torch.full_like(score, C)
            else:
                lam = C / score  # per-feature Î»
            lam = torch.clamp(lam, min=1e-8, max=0.5)

            # Soft-threshold for the L1 component of Ratio Norm
            v_shrunk = soft_thresholding(v, lam / rho)

            # Ratio Norm scaling via cubic solver (skip if ablation: plain L1)
            if use_ratio_norm:
                v_norm_l1 = torch.norm(v_shrunk, p=1)
                v_norm_l2 = safe_norm(v_shrunk)

                if v_norm_l1 > 1e-8 and v_norm_l2 > 1e-8:
                    # FIX: D should be SCALAR for Ratio Norm
                    # The cubic equation ÏÂ³ - Ï - D = 0 derives from global scaling
                    # D = (Î»/Ï) * ||v_shrunk||â?/ ||v_shrunk||âÂ?                    # This is a scalar that scales all features uniformly
                    D_scalar = (
                        (C / rho) * v_norm_l1 / torch.clamp(v_norm_l2**3, min=1e-10)
                    )

                    # Solve for scalar Ï
                    tau_k = solve_cubic_ratio_norm(
                        torch.tensor([D_scalar], device=v.device)
                    )

                    # Apply Ratio Norm scaling to shrunk v
                    zk = tau_k * v_shrunk
                else:
                    # v â?0: all features pruned, keep zeros
                    zk = v_shrunk
            else:
                # Plain L1: just soft-thresholding, no Ratio Norm scaling
                zk = v_shrunk
            # For bounded_gate (raw space ADMM): no projection needed
            # The sigmoid naturally constrains effective gate to [0,1]
            # For unbounded_gate (effective space ADMM): project to valid range
            if not bounded_gate:
                zk = _project_effective_gate(zk)

        # ââ (3) Dual update (once per epoch) ââââââââââââââââââââââââââ
        with torch.no_grad():
            # For bounded_gate: use raw gate; for unbounded: use effective gate
            if bounded_gate:
                uk = uk + gate_param.data - zk
            else:
                gate_value = _effective_gate(gate_param.data)
                uk = uk + gate_value - zk
        # Log convergence diagnostics
        with torch.no_grad():
            if bounded_gate:
                primal_resid = torch.norm(gate_param.data - zk).item()
            else:
                gate_value = _effective_gate(gate_param.data)
                primal_resid = torch.norm(gate_value - zk).item()
            dual_resid = rho * torch.norm(zk - zk_old).item()
            convergence_log.append((epoch, primal_resid, dual_resid))

        # ââ Early Stopping Check ââââââââââââââââââââââââââââââââââââââ
        if use_early_stopping and val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_val_hat = model(x_val)
                    if n_classes > 2:
                        y_val_hat = torch.log_softmax(y_val_hat, dim=1)
                    else:
                        y_val_hat = y_val_hat.reshape(len(y_val_hat))
                    val_loss += criterion(y_val_hat, y_val.float()).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best state
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(
                        f"Early stopping at epoch {warmup_epochs + epoch}, best val loss: {best_val_loss:.4f}"
                    )
                    break

        # ââ Adaptive Ï (Boyd Â§3.4.1) with proper dual rescaling ââââââ
        if epoch > 0 and epoch % rho_update_interval == 0:
            with torch.no_grad():
                gate_value = _effective_gate(gate_param.data)
                r_norm = torch.norm(gate_value - zk).item()
                s_norm = rho * torch.norm(zk - zk_old).item()
                mu_bal = 10.0
                r_n = max(r_norm, 1e-12)
                s_n = max(s_norm, 1e-12)
                rho_old = rho
                if r_n > mu_bal * s_n:
                    rho = min(rho * 2.0, 1e4)
                elif s_n > mu_bal * r_n:
                    # FIX: Lower rho_min for low-dimensional tasks
                    rho_min = (
                        5.0 if n_features < 64 else (20.0 if n_features < 256 else 50.0)
                    )
                    rho = max(rho / 2.0, rho_min)
                # Rescale dual variable when Ï changes (Boyd Â§3.4.1)
                if rho != rho_old:
                    uk = uk * (rho_old / rho)

    # Restore best state if early stopping was triggered
    if use_early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (val loss: {best_val_loss:.4f})")

    model.eval()
    model.convergence_log = convergence_log


def _train_adaptive_input_group(
    model,  # AdaptiveFeatureSelectionMLP or AdaptiveFeatureSelector
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    lr: float = 0.005,
    C: float = 0.05,
    epochs: int = 500,
    warmup_epochs: int = 120,
    batch_size: int = 64,
    rho_init: float = 200.0,
    device: Optional[str] = None,
    use_ratio_norm: bool = True,
    use_admm: bool = True,
    optimizer_type: str = "adam",
    use_early_stopping: bool = False,
    patience: int = 66,
    val_split: float = 0.2,
) -> None:
    """Train an adaptive model with standard training (simplified for adaptive architectures).

    This is a simplified training function for adaptive models that don't have
    the same structure as GatedFeatureSelectionMLP.

    Args:
        model: AdaptiveFeatureSelectionMLP or AdaptiveFeatureSelector
        X_train: Training features
        y_train: Training labels
        n_classes: Number of classes
        lr: Learning rate
        C: Sparsity coefficient (not used in simplified version)
        epochs: Total training epochs
        warmup_epochs: Warmup epochs before ADMM (not used in simplified version)
        batch_size: Batch size
        rho_init: Initial ADMM penalty parameter (not used in simplified version)
        device: Training device
        use_ratio_norm: Use Ratio Norm (not used in simplified version)
        use_admm: Use ADMM (not used in simplified version)
        optimizer_type: 'adam' or 'adagrad'
        use_early_stopping: Enable early stopping
        patience: Early stopping patience
        val_split: Validation split
    """
    # Reuse the full ADMM path when the adaptive model still exposes a
    # global per-feature gate vector compatible with the original solver.
    gate_attr = getattr(model, "gate", None)
    if isinstance(gate_attr, nn.Parameter) and hasattr(model, "first_linear"):
        _train_input_group(
            model,
            X_train,
            y_train,
            n_classes,
            lr=lr,
            C=C,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            batch_size=batch_size,
            rho_init=rho_init,
            device=device,
            use_ratio_norm=use_ratio_norm,
            use_admm=use_admm,
            optimizer_type=optimizer_type,
            use_early_stopping=use_early_stopping,
            patience=patience,
            val_split=val_split,
            n_features=X_train.shape[1],
        )
        return

    N = len(X_train)
    n_features = X_train.shape[1]

    # Validation split for early stopping
    if use_early_stopping and val_split > 0:
        n_val = int(len(X_train) * val_split)
        idx = np.random.permutation(len(X_train))
        X_val, y_val = X_train[idx[:n_val]], y_train[idx[:n_val]]
        X_train, y_train = X_train[idx[n_val:]], y_train[idx[n_val:]]
    else:
        X_val, y_val = None, None

    dataset = TrainingSet(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if X_val is not None:
        val_dataset = TrainingSet(X_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        val_loader = None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Optimizer choice
    if optimizer_type == "adagrad":
        opt_class = torch.optim.Adagrad
    else:
        opt_class = torch.optim.Adam

    optimizer = opt_class(
        list(model.parameters()),
        lr=lr,
        weight_decay=1e-3,
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss() if n_classes <= 2 else torch.nn.CrossEntropyLoss()
    )

    convergence_log = []
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            # Use .float()/.long() methods for proper dtype conversion
            batch_X = batch_X.float().to(device)
            batch_y = (
                batch_y.float().to(device)
                if n_classes <= 2
                else batch_y.long().to(device)
            )

            optimizer.zero_grad()
            logits = model(batch_X)

            if n_classes <= 2:
                batch_y = batch_y.view(-1, 1)
                loss = criterion(logits, batch_y)
            else:
                loss = criterion(logits, batch_y.squeeze().long())

            loss.backward()
            optimizer.step()

        # Validation and early stopping
        if use_early_stopping and val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.float().to(device)
                    batch_y = (
                        batch_y.float().to(device)
                        if n_classes <= 2
                        else batch_y.long().to(device)
                    )
                    logits = model(batch_X)
                    if n_classes <= 2:
                        batch_y = batch_y.view(-1, 1)
                    val_loss += criterion(logits, batch_y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    break

    # Restore best model
    if use_early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    model.convergence_log = convergence_log


def _extract_feature_importance(model, X_train: np.ndarray) -> np.ndarray:
    """Feature importance extraction.

    For GatedFeatureSelectionMLP: returns |gate| values directly.
    For AdaptiveFeatureSelector/MLP: uses get_gate_values() method.
    For FeatureSelectionMLP: WANDA-style  score[j] = sum_i |Wâ[i, j]| * ||a_j||â?    """
    # If model has get_gate_values method (AdaptiveFeatureSelector, AdaptiveFeatureSelectionMLP)
    if hasattr(model, "get_gate_values"):
        try:
            x_sample = torch.as_tensor(
                X_train[: min(1024, len(X_train))], dtype=torch.float32
            )
            gate_values = model.get_gate_values(x_sample)
        except TypeError:
            gate_values = model.get_gate_values()
        return gate_values.detach().abs().cpu().numpy()

    # If model has a learnable gate, use it directly
    if hasattr(model, "gate"):
        if getattr(model, "bounded_gate", False):
            return torch.sigmoid(model.gate.data).cpu().numpy()
        return model.gate.data.abs().cpu().numpy()
    # Capture activation at the first Linear layer
    first_linear = model.first_linear
    activation = None

    def hook_fn(module, inp, out):
        nonlocal activation
        activation = inp[0].detach()

    handle = first_linear.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        x_sample = torch.FloatTensor(X_train[: min(512, len(X_train))])
        # Move to same device as model
        device = next(model.parameters()).device
        model(x_sample.to(device))
    handle.remove()

    W = first_linear.weight.data  # (out_features, in_features)
    act_norm = torch.norm(activation, p=2, dim=0)  # (in_features,)

    # WANDA score per input feature: sum over output neurons of |W_ij| * ||a_j||
    wanda_per_feature = (torch.abs(W) * act_norm.unsqueeze(0)).sum(
        dim=0
    )  # (in_features,)
    return wanda_per_feature.cpu().numpy()


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def _predict_proba(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.FloatTensor(X).to(device)
        logits = model(x_t)
        if n_classes <= 2:
            proba = torch.sigmoid(logits).squeeze().cpu().numpy()
        else:
            proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba


# ---------------------------------------------------------------------------
# Public API: adaptive input-group framework
# ---------------------------------------------------------------------------


def run_adaptive_input_group(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    *,
    model_type: str = "adaptive_gate",
    hp_overrides: Optional[dict] = None,
    seed: Optional[int] = None,
):
    """Train an adaptive feature-selection model and return FS results.

    Returns:
        (y_train_hat, y_hat, scores, scores2)
    """
    if not ADAPTIVE_MODELS_AVAILABLE:
        raise RuntimeError(
            "Adaptive models are unavailable. Ensure adaptive_architecture.py is importable."
        )

    n_features = X_train.shape[1]
    hp = {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "batch_size": 64,
        "rho_init": 200.0,
        "optimizer_type": "adam",
        "use_early_stopping": True,
        "patience": 66,
        "val_split": 0.2,
        # model kwargs
        "feat_drop": 0.6,
        "dropout": 0.043 if model_type == "adaptive_mlp" else 0.0,
        "gaussian_noise": 0.0,
    }
    if hp_overrides:
        hp.update(hp_overrides)

    scaler = _Scaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    _seed = seed if seed is not None else hash(tuple(y_train[:20].tolist())) % 2**31
    torch.manual_seed(_seed)

    model_kwargs = {
        "feat_drop": hp.get("feat_drop", 0.6),
        "dropout": hp.get("dropout", 0.0),
        "gaussian_noise": hp.get("gaussian_noise", 0.0),
    }
    model = create_adaptive_model(
        n_features=n_features,
        n_classes=n_classes,
        model_type=model_type,
        **model_kwargs,
    )

    _train_adaptive_input_group(
        model,
        X_train_s,
        y_train,
        n_classes,
        lr=hp["lr"],
        C=hp["C"],
        epochs=hp["epochs"],
        warmup_epochs=hp["warmup_epochs"],
        batch_size=hp["batch_size"],
        rho_init=hp["rho_init"],
        optimizer_type=hp["optimizer_type"],
        use_early_stopping=hp["use_early_stopping"],
        patience=hp["patience"],
        val_split=hp["val_split"],
    )

    scores = _extract_feature_importance(model, X_train_s)
    scores2 = scores
    y_train_hat = _predict_proba(model, X_train_s, n_classes)
    y_hat = _predict_proba(model, X_test_s, n_classes)
    return y_train_hat, y_hat, scores, scores2


# ---------------------------------------------------------------------------
# Public API: admm_input_group only
# ---------------------------------------------------------------------------


def run_admm_input_group(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    *,
    use_ratio_norm: bool = True,
    use_admm: bool = True,
    bounded_gate: bool = False,
    hp_overrides: Optional[dict] = None,
    seed: Optional[int] = None,
):
    """Train an MLP with admm_input_group and return FS results.

    Implements three strategies:
      1. Warm start (ADMM only): initialise first layer from L1-logistic
      2. Standardisation: z-score input features
      3. Adaptive rho: Boyd's primal-dual balancing (inside training loop)

    Returns:
        (y_train_hat, y_hat, scores, scores2)
    """
    # Default hyperparameters for admm_input_group
    # Updated 2026-03-11:
    #   - feat_drop tuned from 0.7 to 0.6 (improves ring avg_best_k from 0.12 to 0.31)
    #   - gaussian_noise disabled by default (can be enabled via hp_overrides)
    #   - rho_init scaled by feature dimension (avoids over-constraint in low-dim tasks)
    n_features = X_train.shape[1]

    # FIX: Scale rho_init with feature dimension
    # Low-dimensional tasks (m < 256) need smaller rho to avoid over-constraint
    # High-dimensional tasks (m >= 256) can use larger rho for faster convergence
    if n_features < 64:
        rho_init_default = 20.0  # Very small for very low dimensions
    elif n_features < 256:
        rho_init_default = 50.0  # Small for low dimensions
    elif n_features < 512:
        rho_init_default = 100.0  # Medium for medium dimensions
    else:
        rho_init_default = 200.0  # Original value for high dimensions

    hp = {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "batch_size": 64,
        "latent_size": DEFAULT_SADMM_LATENT_SIZE,
        "n_hidden_layers": DEFAULT_SADMM_HIDDEN_LAYERS,
        "dropout": DEFAULT_SADMM_DROPOUT,
        "activation": DEFAULT_SADMM_ACTIVATION,
        "feat_drop": 0.6,  # Tuned from 0.7
        "gaussian_noise": 0.0,  # Optional; set >0 to enable input noise
        "column_normalize_first_layer": False,
        "rho_init": rho_init_default,  # Scaled by dimension
        "warm_start": False,
        "optimizer_type": "adam",
        "use_early_stopping": False,
        "patience": 66,
        "val_split": 0.2,
    }
    if hp_overrides:
        hp.update(hp_overrides)

    # ---- Strategy 2: Standardise features ----
    scaler = _Scaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Model selection ----
    # Fix torch seed per call for reproducibility across CV folds
    _seed = seed if seed is not None else hash(tuple(y_train[:20].tolist())) % 2**31
    torch.manual_seed(_seed)
    _use_ratio = use_ratio_norm
    model = GatedFeatureSelectionMLP(
        input_size=n_features,
        n_classes=n_classes,
        latent_size=hp["latent_size"],
        n_hidden_layers=hp["n_hidden_layers"],
        gaussian_noise=hp.get("gaussian_noise", 0.0),
        dropout=hp["dropout"],
        feat_drop=hp.get("feat_drop", 0.6),
        activation=hp["activation"],
        bounded_gate=bounded_gate,
        column_normalize_first_layer=hp.get("column_normalize_first_layer", False),
    )

    _train_input_group(
        model,
        X_train_s,
        y_train,
        n_classes,
        lr=hp["lr"],
        C=hp["C"],
        epochs=hp["epochs"],
        warmup_epochs=hp.get("warmup_epochs", 120),
        batch_size=hp.get("batch_size", 64),
        rho_init=hp.get("rho_init", 200.0),
        use_ratio_norm=_use_ratio,
        use_admm=use_admm,
        n_features=n_features,  # Pass for rho scaling
        optimizer_type=hp.get("optimizer_type", "adam"),
        use_early_stopping=hp.get("use_early_stopping", False),
        patience=hp.get("patience", 66),
        val_split=hp.get("val_split", 0.2),
    )

    scores = _extract_feature_importance(model, X_train_s)
    scores2 = scores

    y_train_hat = _predict_proba(model, X_train_s, n_classes)
    y_hat = _predict_proba(model, X_test_s, n_classes)

    return y_train_hat, y_hat, scores, scores2


# ---------------------------------------------------------------------------
# Default hyperparameters (for backward compatibility with old _HPARAMS)
# ---------------------------------------------------------------------------

DEFAULT_HPARAMS = {
    "admm_input_group": {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "feat_drop": 0.6,
        "rho_init": 200.0,  # Will be scaled by dimension in run_admm_input_group
        "warm_start": False,
    },
    # Aliases for backward compatibility
    "lasso_input_group": {
        "lr": 0.005,
        "C": 0.05,
        "epochs": 500,
        "warmup_epochs": 120,
        "feat_drop": 0.6,
        "rho_init": 200.0,
        "warm_start": False,
    },
}

# Alias for backward compatibility with old admm_lasso_wrapper.py
_HPARAMS = DEFAULT_HPARAMS


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

__all__ = [
    "FeatureSelectionMLP",
    "GatedFeatureSelectionMLP",
    "ColumnNormalizedLinear",
    "_Scaler",
    "_train_input_group",
    "_train_adaptive_input_group",  # For adaptive models
    "_extract_feature_importance",
    "_predict_proba",
    "run_adaptive_input_group",
    "run_admm_input_group",
    "DEFAULT_HPARAMS",
    "_HPARAMS",  # Backward compatibility alias
    # Adaptive models (if available)
    "ADAPTIVE_MODELS_AVAILABLE",
    # Re-export utilities for convenience
    "soft_thresholding",
    "safe_norm",
    "safe_cbrt",
    "solve_cubic_ratio_norm",
    "solve_cubic_paper",
]
