# -*- coding: utf-8 -*-
"""
ADMM / Lasso feature-selection wrappers for the Feature-Selection-Benchmark.

Six methods are provided (3 regularizers x 2 granularities kept simple):
  admm_global, admm_layer, admm_neuron
  lasso_global, lasso_layer, lasso_neuron

Each wrapper:
  1. Builds a small MLP identical to the benchmark's ``Model`` architecture.
  2. Trains it with the corresponding ADMM or Lasso optimizer from the
     ``optimizer/`` package (the pruning codebase).
  3. After training, derives *feature importance scores* as the L2 norm
     of each input-feature column in the first Linear layer's weight matrix:
       score[j] = ||W_1[:, j]||_2
     Features whose incoming weights are pruned to zero receive a score of 0.
  4. Returns ``(y_train_hat, y_hat, scores, scores2)`` as required by
     ``run_fs_method`` in ``src/core.py``.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.nn_wrapper import GaussianNoise, init_weights
from src.utils import TrainingSet, TestSet

# ---------------------------------------------------------------------------
# Make the pruning-optimizer package importable
# ---------------------------------------------------------------------------
_PRUNING_ROOT = str(Path(__file__).resolve().parents[1].parent)  # ..../NEW_Pruning_20251110
if _PRUNING_ROOT not in sys.path:
    sys.path.insert(0, _PRUNING_ROOT)

from optimizer.ADMM_global import ADMM_Adam_global
from optimizer.ADMM_layer import ADMM_Adam_layer
from optimizer.ADMM_neuron import ADMM_Adam_neuron
from optimizer.ADMM_input_group import ADMM_Input_Group
from optimizer.lasso_global import Lasso_global
from optimizer.lasso_layer import Lasso_layer
from optimizer.lasso_neuron import Lasso_neuron
from optimizer.utils import (
    soft_thresholding,
    solve_cubic_ratio_norm,
    safe_norm,
    safe_cbrt,
)


# ---------------------------------------------------------------------------
# MLP model (mirrors the benchmark's ``Model`` but simplified for FS)
# ---------------------------------------------------------------------------

class FeatureSelectionMLP(nn.Module):
    """Simple MLP for tabular classification, compatible with sparsity optimizers."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = 58,
        n_hidden_layers: int = 5,
        gaussian_noise: float = 0.0,
        dropout: float = 0.04,
        activation: str = "mish",
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
    def first_linear(self) -> nn.Linear:
        for m in self.layers:
            if isinstance(m, nn.Linear):
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

    Architecture is intentionally *smaller* than ``FeatureSelectionMLP``
    (2 hidden layers, 32 units) to further control capacity.
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        latent_size: int = 32,
        n_hidden_layers: int = 2,
        feat_drop: float = 0.7,
        activation: str = "mish",
    ):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(input_size))
        self.feat_drop = feat_drop
        n_out = 1 if n_classes <= 2 else n_classes

        layers: list[nn.Module] = []
        for k in range(n_hidden_layers):
            in_dim = input_size if k == 0 else latent_size
            layers.append(nn.Linear(in_dim, latent_size))
            if activation == "mish":
                layers.append(nn.Mish())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Mish())
        layers.append(nn.Linear(latent_size, n_out))
        self.layers = nn.Sequential(*layers)
        # Use PyTorch default init (kaiming_uniform_ a=sqrt(5)) — empirically
        # better for gate-based feature selection than the custom init_weights.

    @property
    def first_linear(self) -> nn.Linear:
        for m in self.layers:
            if isinstance(m, nn.Linear):
                return m
        raise RuntimeError("No Linear layer found")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate
        if self.training and self.feat_drop > 0:
            mask = (torch.rand(g.shape, device=x.device) > self.feat_drop).float()
            g = g * mask / (1.0 - self.feat_drop + 1e-8)
        return self.layers(x * g)


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
# Strategy 1: warm-start ADMM from sklearn L1-logistic regression
# ---------------------------------------------------------------------------

def _warm_start_from_lasso(
    model: FeatureSelectionMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    C_lasso: float = 1.0,
) -> None:
    """Initialise the first Linear layer from sklearn L1-logistic regression.

    This gives ADMM a good starting point in the correct basin,
    avoiding the high-dimensional random-initialisation curse.
    """
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=C_lasso,
        max_iter=200,
        tol=1e-3,
        multi_class="auto",
        random_state=42,
    )
    lr_model.fit(X_train, y_train)

    first = model.first_linear
    coef = lr_model.coef_            # (n_classes_or_1, n_features)
    intercept = lr_model.intercept_  # (n_classes_or_1,)

    with torch.no_grad():
        coef_t = torch.FloatTensor(coef)
        out_f, in_f = first.weight.shape

        if coef_t.shape[0] == 1:
            init_w = coef_t.expand(out_f, -1).clone()
            init_w += 0.01 * torch.randn_like(init_w)
        else:
            repeats = (out_f + coef_t.shape[0] - 1) // coef_t.shape[0]
            init_w = coef_t.repeat(repeats, 1)[:out_f]
            init_w += 0.01 * torch.randn_like(init_w)

        first.weight.copy_(init_w)

        bias_t = torch.FloatTensor(intercept)
        if bias_t.shape[0] == 1:
            first.bias.copy_(bias_t.expand(out_f) + 0.001 * torch.randn(out_f))
        else:
            repeats = (out_f + bias_t.shape[0] - 1) // bias_t.shape[0]
            first.bias.copy_(bias_t.repeat(repeats)[:out_f] + 0.001 * torch.randn(out_f))


# ---------------------------------------------------------------------------
# Strategy 3: Adaptive rho (Boyd et al. 2011, section 3.4.1)
# ---------------------------------------------------------------------------

def _adaptive_rho_update(
    opt,
    mu: float = 10.0,
    tau_incr: float = 2.0,
    tau_decr: float = 2.0,
    rho_min: float = 50.0,
    rho_max: float = 1e4,
) -> None:
    """Adjust rho based on primal vs dual residual norms.

    Boyd's rule:
      if ||r_k|| > mu * ||s_k||  --> rho *= tau_incr
      if ||s_k|| > mu * ||r_k||  --> rho /= tau_decr
    Also rescale dual variables to keep ADMM consistent.
    """
    r_norm = getattr(opt, 'r_norm', None)
    s_norm = getattr(opt, 's_norm', None)
    if r_norm is None or s_norm is None:
        return

    r_norm = max(r_norm, 1e-12)
    s_norm = max(s_norm, 1e-12)
    rho_old = opt.rho

    if r_norm > mu * s_norm:
        opt.rho = min(rho_old * tau_incr, rho_max)
    elif s_norm > mu * r_norm:
        opt.rho = max(rho_old / tau_decr, rho_min)
    else:
        return  # no change

    # Rescale dual variables for consistency: v,w *= rho_old / rho_new
    scale = rho_old / opt.rho
    if hasattr(opt, 'vk') and hasattr(opt, 'wk'):
        if isinstance(opt.vk, list):
            for vt, wt in zip(opt.vk, opt.wk):
                vt.mul_(scale)
                wt.mul_(scale)
        else:
            # Global: vk/wk are parameter lists, need vector ops
            vk_vec = parameters_to_vector(opt.vk)
            wk_vec = parameters_to_vector(opt.wk)
            vector_to_parameters(vk_vec * scale, opt.vk)
            vector_to_parameters(wk_vec * scale, opt.wk)


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
# Generic training loop
# ---------------------------------------------------------------------------

def _train_with_optimizer(
    model: FeatureSelectionMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    optimizer_cls,
    is_admm: bool,
    lr: float = 0.005,
    C: float = 0.08,
    epochs: int = 100,
    batch_size: int = 64,
    score_refresh_interval: int = 10,
    rho_update_interval: int = 5,
    device: str = "cpu",
) -> None:
    """Train *model* in-place using the given sparsity optimizer."""

    N = len(X_train)
    dataset = TrainingSet(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    params = list(model.parameters())
    zeros = [torch.zeros_like(p) for p in params]
    score_bufs = [torch.ones_like(p) for p in params]

    # Build optimizer
    if is_admm:
        opt = optimizer_cls(
            params,
            lr=lr,
            N=N,
            C=C,
            vk=[z.clone() for z in zeros],
            wk=[z.clone() for z in zeros],
            yk=[p.clone().detach() for p in params],
            zk=[p.clone().detach() for p in params],
            score=score_bufs,
        )
    else:
        opt = optimizer_cls(
            params,
            lr=lr,
            N=N,
            C=C,
            vk=[z.clone() for z in zeros],
            zk=[p.clone().detach() for p in params],
            score=score_bufs,
        )

    if n_classes <= 2:
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        criterion = nn.NLLLoss(reduction="mean")

    model.train()
    for epoch in range(epochs):
        # Refresh WANDA scores periodically
        if epoch % score_refresh_interval == 0:
            model.eval()
            sample_x = torch.FloatTensor(X_train[:min(256, N)])
            new_scores = compute_mlp_wanda_scores(model, sample_x)
            for sb, ns in zip(score_bufs, new_scores):
                sb.copy_(ns)
            model.train()

        # Strategy 3: adaptive rho every rho_update_interval epochs
        if is_admm and epoch > 0 and epoch % rho_update_interval == 0:
            _adaptive_rho_update(opt)

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            opt.zero_grad()
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
            opt.step()

    model.eval()


# ---------------------------------------------------------------------------
# ADMM-Gate training loop for Input-Group feature selection
# ---------------------------------------------------------------------------

def _train_input_group(
    model,  # GatedFeatureSelectionMLP
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    lr: float = 0.005,
    C: float = 0.1,         # Ratio Norm sparsity coefficient
    epochs: int = 500,
    warmup_epochs: int = 120,
    batch_size: int = 64,
    rho_init: float = 200.0,
    rho_update_interval: int = 5,
    score_refresh_interval: int = 10,
    device: str = "cpu",
    use_ratio_norm: bool = True,   # False → plain L1 proximal (ablation)
    use_admm: bool = True,         # False → proximal gradient (ablation)
) -> None:
    """Train a GatedFeatureSelectionMLP with Linearized ADMM + Ratio Norm.

    **Phase 1 — Warm-up** (epoch 0 … warmup_epochs-1):
        All parameters trained with Adam.  Feature dropout (in the model)
        prevents memorisation, letting the MLP learn the true signal.

    **Phase 2 — Linearized ADMM pruning** (epoch warmup_epochs … epochs-1):
        Uses 2-variable ADMM splitting:  min L(θ,g) + λ·R(z)  s.t. g = z
        where R(z) = ‖z‖₁/‖z‖₂ is the Ratio Norm (scale-invariant).

        Per epoch:
          1. **g-step**: Adam on ALL params (θ AND g) with augmented loss
             L(θ,g) + (ρ/2)‖g − z + u‖²   → gate has full gradient dynamics
          2. **z-step**: z = prox_{λ/ρ · RatioNorm}(g + u)
             → cubic τ³ − τ − D = 0 for scaling + soft-thresh for sparsity
          3. **Dual update**: u ← u + g − z
          4. **Adaptive ρ**: Boyd §3.4.1 with proper dual rescaling

        Score s_j = ‖W₁[:,j]‖₂ gives importance-adaptive thresholds
        λ_j = C/s_j, so the Ratio Norm amplifies signal features and
        suppresses noise — an effect absent in plain L1.
    """
    N = len(X_train)
    dataset = TrainingSet(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if n_classes <= 2:
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        criterion = nn.NLLLoss(reduction="mean")

    # ==================================================================
    # Phase 1: Warm-up — Adam on ALL params (feat-drop active in model)
    # ==================================================================
    opt_warmup = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

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
    # Phase 2: Linearized ADMM — gate in Adam + Ratio Norm z-step
    #
    #   min_{g,θ}  L(θ, g)  +  λ · R(z)           (R = Ratio Norm)
    #   s.t.  g = z
    #
    #   Augmented Lagrangian (scaled form):
    #     L_ρ = L(θ, g) + (ρ/2) ‖g − z + u‖²
    #
    #   Per epoch:
    #     (1) g-step:  Adam on (θ, g) with augmented loss
    #                  → gate gets full gradient dynamics
    #     (2) z-step:  z = prox_{λ/ρ · RatioNorm}(g + u)
    #                  → Ratio Norm proximal (cubic solver) for sparsity
    #     (3) dual:    u ← u + g − z
    #
    #   Score s_j = ‖W₁[:,j]‖₂ gives importance-adaptive thresholds.
    # ==================================================================
    gate_param = model.gate  # shape (m,)
    m = gate_param.shape[0]

    # ADMM buffers (scaled dual form: u = λ/ρ)
    zk = gate_param.data.clone()
    uk = torch.zeros(m, device=device)  # scaled dual variable
    rho = rho_init

    # Score = first-layer column norms (importance per feature)
    with torch.no_grad():
        W1 = model.first_linear.weight  # (out, in=m)
        score = torch.norm(W1, p=2, dim=0) + 1e-8  # (m,)

    # Adam on ALL parameters including gate (gate gets gradient dynamics)
    opt_all = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    prune_epochs = epochs - warmup_epochs

    # ── Ablation: proximal gradient (no ADMM) ────────────────────────
    if not use_admm:
        opt_prox = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        for epoch in range(prune_epochs):
            with torch.no_grad():
                W1 = model.first_linear.weight
                score = torch.norm(W1, p=2, dim=0) + 1e-8

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
                g = gate_param.data
                lam = C / score
                lam = torch.clamp(lam, min=1e-6, max=0.5)
                alpha = lam * lr  # proximal step size
                g_shrunk = soft_thresholding(g, alpha)
                if use_ratio_norm and torch.norm(g_shrunk, p=1) > 1e-8:
                    v_l2 = safe_norm(g_shrunk)
                    mu = C * torch.norm(g_shrunk, p=1) / N
                    D_k = (mu * score * score) / (1.0 * torch.clamp(v_l2 ** 3, min=1e-10))
                    tau_k = solve_cubic_ratio_norm(D_k)
                    g_shrunk = tau_k * g_shrunk
                gate_param.data.copy_(g_shrunk)
        model.eval()
        return

    # ── ADMM path (default) ───────────────────────────────────────────
    convergence_log = []  # stores (epoch, primal_resid, dual_resid) for diagnostics
    for epoch in range(prune_epochs):
        # Refresh score every epoch (column norms evolve with training)
        with torch.no_grad():
            W1 = model.first_linear.weight
            score = torch.norm(W1, p=2, dim=0) + 1e-8

        # ── (1) g-step: Adam on augmented loss over all mini-batches ──
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
            admm_penalty = (rho / 2.0) * torch.sum(
                (gate_param - zk.detach() + uk.detach()) ** 2
            )
            total_loss = data_loss + admm_penalty

            total_loss.backward()
            opt_all.step()

        # ── (2) z-step: Ratio Norm proximal (once per epoch) ──────────
        with torch.no_grad():
            zk_old = zk.clone()
            v = gate_param.data + uk  # input to proximal operator

            # --- Ratio Norm proximal on v ---
            # Ratio Norm R(z) = ‖z‖₁ / ‖z‖₂  (scale-invariant)
            # prox_{λ/ρ · R}(v) decomposes into:
            #   direction:  solve  τ³ − τ − D = 0  for scaling
            #   shrinkage:  soft-threshold for L1 component
            #
            # Importance-adaptive λ_j = C / s_j  (strong on noise,
            # weak on signal — threshold ~0.05 scale, matching proximal L1)
            lam = C / score                    # per-feature λ
            lam = torch.clamp(lam, min=1e-6, max=0.5)

            # Soft-threshold for the L1 component of Ratio Norm
            v_shrunk = soft_thresholding(v, lam / rho)

            # Ratio Norm scaling via cubic solver (skip if ablation: plain L1)
            if use_ratio_norm:
                v_norm_l1 = torch.norm(v_shrunk, p=1)
                v_norm_l2 = safe_norm(v_shrunk)

                if v_norm_l1 > 1e-8 and v_norm_l2 > 1e-8:
                    # D_k for cubic: measures trade-off between L1 shrinkage
                    # and L2 scaling.  Per-element D_k via score weighting.
                    mu = C * v_norm_l1 / N
                    D_k = (mu * score * score) / (rho * torch.clamp(v_norm_l2 ** 3, min=1e-10))
                    tau_k = solve_cubic_ratio_norm(D_k)

                    # Apply Ratio Norm scaling to shrunk v
                    zk = tau_k * v_shrunk
                else:
                    # v ≈ 0: all features pruned, keep zeros
                    zk = v_shrunk
            else:
                # Plain L1: just soft-thresholding, no Ratio Norm scaling
                zk = v_shrunk

        # ── (3) Dual update (once per epoch) ──────────────────────────
        with torch.no_grad():
            uk = uk + gate_param.data - zk

        # Log convergence diagnostics
        with torch.no_grad():
            primal_resid = torch.norm(gate_param.data - zk).item()
            dual_resid = rho * torch.norm(zk - zk_old).item()
            convergence_log.append((epoch, primal_resid, dual_resid))

        # ── Adaptive ρ (Boyd §3.4.1) with proper dual rescaling ──────
        if epoch > 0 and epoch % rho_update_interval == 0:
            with torch.no_grad():
                r_norm = torch.norm(gate_param.data - zk).item()
                s_norm = rho * torch.norm(zk - zk_old).item()
                mu_bal = 10.0
                r_n = max(r_norm, 1e-12)
                s_n = max(s_norm, 1e-12)
                rho_old = rho
                if r_n > mu_bal * s_n:
                    rho = min(rho * 2.0, 1e4)
                elif s_n > mu_bal * r_n:
                    rho = max(rho / 2.0, 50.0)
                # Rescale dual variable when ρ changes (Boyd §3.4.1)
                if rho != rho_old:
                    uk = uk * (rho_old / rho)

    model.eval()
    model.convergence_log = convergence_log

def _extract_feature_importance(
    model, X_train: np.ndarray
) -> np.ndarray:
    """Feature importance extraction.

    For GatedFeatureSelectionMLP: returns |gate| values directly.
    For FeatureSelectionMLP: WANDA-style  score[j] = sum_i |W₁[i, j]| * ||a_j||₂
    """
    # If model has a learnable gate, use it directly
    if hasattr(model, 'gate'):
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
        model(torch.FloatTensor(X_train[:min(512, len(X_train))]))
    handle.remove()

    W = first_linear.weight.data           # (out_features, in_features)
    act_norm = torch.norm(activation, p=2, dim=0)  # (in_features,)

    # WANDA score per input feature: sum over output neurons of |W_ij| * ||a_j||
    wanda_per_feature = (torch.abs(W) * act_norm.unsqueeze(0)).sum(dim=0)  # (in_features,)
    return wanda_per_feature.cpu().numpy()


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def _predict_proba(
    model: FeatureSelectionMLP, X: np.ndarray, n_classes: int
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.FloatTensor(X)
        logits = model(x_t)
        if n_classes <= 2:
            proba = torch.sigmoid(logits).squeeze().cpu().numpy()
        else:
            proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba


# ---------------------------------------------------------------------------
# Public API: one function per method
# ---------------------------------------------------------------------------

_OPTIMIZER_MAP = {
    "admm_global": (ADMM_Adam_global, True),
    "admm_layer": (ADMM_Adam_layer, True),
    "admm_neuron": (ADMM_Adam_neuron, True),
    "admm_input_group": (ADMM_Input_Group, True),   # Plan A: column-group
    "lasso_input_group": (ADMM_Input_Group, True),   # L1 ablation (no ratio norm)
    "lasso_global": (Lasso_global, False),
    "lasso_layer": (Lasso_layer, False),
    "lasso_neuron": (Lasso_neuron, False),
}

# Hyperparameters tuned per variant
_HPARAMS = {
    "admm_global":       {"lr": 0.005, "C": 0.08, "epochs": 100, "warm_start": True},
    "admm_layer":        {"lr": 0.005, "C": 0.08, "epochs": 100, "warm_start": True},
    "admm_neuron":       {"lr": 0.005, "C": 0.08, "epochs": 100, "warm_start": True},
    "admm_input_group":  {"lr": 0.005, "C": 0.05, "epochs": 500, "warmup_epochs": 120, "feat_drop": 0.7, "rho_init": 200.0, "warm_start": False},
    "lasso_input_group": {"lr": 0.005, "C": 0.05, "epochs": 500, "warmup_epochs": 120, "feat_drop": 0.7, "rho_init": 200.0, "warm_start": False},
    "lasso_global":      {"lr": 0.005, "C": 0.08, "epochs": 100, "warm_start": False},
    "lasso_layer":       {"lr": 0.005, "C": 0.08, "epochs": 100, "warm_start": False},
    "lasso_neuron":      {"lr": 0.005, "C": 0.08, "epochs": 100, "warm_start": False},
}


def run_admm_lasso_fs(
    method_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    *,
    use_ratio_norm: bool = True,
    use_admm: bool = True,
    hp_overrides: dict = None,
    seed: int = None,
):
    """Train an MLP with the specified ADMM/Lasso optimizer and return FS results.

    Implements three strategies:
      1. Warm start (ADMM only): initialise first layer from L1-logistic
      2. Standardisation: z-score input features
      3. Adaptive rho: Boyd's primal-dual balancing (inside training loop)

    Returns:
        (y_train_hat, y_hat, scores, scores2)
    """
    opt_cls, is_admm = _OPTIMIZER_MAP[method_name]
    hp = dict(_HPARAMS[method_name])  # copy so overrides don't mutate global
    if hp_overrides:
        hp.update(hp_overrides)
    n_features = X_train.shape[1]

    # ---- Strategy 2: Standardise features ----
    scaler = _Scaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Model selection ----
    if method_name in ("admm_input_group", "lasso_input_group"):
        # Fix torch seed per call for reproducibility across CV folds
        _seed = seed if seed is not None else hash(tuple(y_train[:20].tolist())) % 2**31
        torch.manual_seed(_seed)
        _use_ratio = use_ratio_norm if method_name == "admm_input_group" else False
        model = GatedFeatureSelectionMLP(
            input_size=n_features,
            n_classes=n_classes,
            latent_size=32,
            n_hidden_layers=2,
            feat_drop=hp.get("feat_drop", 0.7),
            activation="mish",
        )
        _train_input_group(
            model, X_train_s, y_train, n_classes,
            lr=hp["lr"], C=hp["C"], epochs=hp["epochs"],
            warmup_epochs=hp.get("warmup_epochs", 120),
            rho_init=hp.get("rho_init", 200.0),
            use_ratio_norm=_use_ratio,
            use_admm=use_admm,
        )
    else:
        model = FeatureSelectionMLP(
            input_size=n_features,
            n_classes=n_classes,
            latent_size=58,
            n_hidden_layers=5,
            gaussian_noise=0.0,
            dropout=0.0,
            activation="mish",
        )

        # ---- Strategy 1: Warm start from Lasso (ADMM only) ----
        if hp.get("warm_start", False):
            _warm_start_from_lasso(model, X_train_s, y_train, n_classes)

        # ---- Standard optimizer path ----
        _train_with_optimizer(
            model, X_train_s, y_train, n_classes, opt_cls, is_admm,
            lr=hp["lr"], C=hp["C"], epochs=hp["epochs"],
        )

    scores = _extract_feature_importance(model, X_train_s)
    scores2 = scores

    y_train_hat = _predict_proba(model, X_train_s, n_classes)
    y_hat = _predict_proba(model, X_test_s, n_classes)

    return y_train_hat, y_hat, scores, scores2
