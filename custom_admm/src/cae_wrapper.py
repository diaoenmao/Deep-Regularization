"""
CAE-style (Concrete Autoencoder) Feature Selection Wrapper.

NOTE: This is a PyTorch proxy implementation, not the original Keras CAE from:
  Balin et al., "Concrete Autoencoders for Differentiable Feature Selection"
  https://github.com/mfbalin/Concrete-Autoencoders

The original Keras implementation depends on legacy backend calls (K.set_learning_phase,
K.update, K.in_train_phase, K.random_uniform) that are absent in current Keras versions.

This proxy uses the same Gumbel-softmax selector mechanism with a comparable
prediction head architecture (2 hidden layers, 32 units, dropout).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class _ConcreteSelectorNet(nn.Module):
    def __init__(self, n_features: int, k: int, n_classes: int = 2):
        super().__init__()
        self.n_features = n_features
        self.k = k
        self.n_classes = n_classes
        self.logits = nn.Parameter(torch.zeros(k, n_features))
        nn.init.xavier_uniform_(self.logits)
        # Match original CAE architecture: GaussianNoise -> Dense -> Dropout -> LeakyReLU
        self.backbone = nn.Sequential(
            nn.Linear(k, 32),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 32),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1 if n_classes <= 2 else n_classes),
        )

    def _selector_matrix(self, temperature: float, training: bool) -> torch.Tensor:
        if training:
            uniform = torch.rand_like(self.logits).clamp_min(1e-8)
            noise = -torch.log(-torch.log(uniform))
            return torch.softmax((self.logits + noise) / temperature, dim=1)
        indices = torch.argmax(self.logits, dim=1)
        return torch.nn.functional.one_hot(indices, num_classes=self.n_features).float()

    def feature_scores(self) -> torch.Tensor:
        return torch.softmax(self.logits.detach(), dim=1).sum(dim=0)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        selectors = self._selector_matrix(temperature, self.training)
        selected = x @ selectors.t()
        return self.backbone(selected)


def _predict_proba(
    model: _ConcreteSelectorNet, X: np.ndarray, device: str
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        if logits.shape[1] == 1:
            return torch.sigmoid(logits[:, 0]).cpu().numpy()
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def _train_single_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k: int,
    n_classes: int,
    seed: int,
    max_epochs: int,
    patience: int,
    lr: float,
    device: str,
) -> _ConcreteSelectorNet:
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
        )
    except ValueError:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
        )

    model = _ConcreteSelectorNet(X_train.shape[1], k, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() if n_classes <= 2 else nn.CrossEntropyLoss()

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32, device=device)
    y_fit_t = torch.tensor(
        y_fit,
        dtype=torch.float32 if n_classes <= 2 else torch.long,
        device=device,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(
        y_val,
        dtype=torch.float32 if n_classes <= 2 else torch.long,
        device=device,
    )

    batch_size = min(128, max(32, len(X_fit) // 8))
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        temperature = max(0.1, 10.0 * (0.95**epoch))
        order = np.random.permutation(len(X_fit))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = X_fit_t[idx]
            yb = y_fit_t[idx]
            logits = model(xb, temperature=temperature)
            loss = (
                criterion(logits[:, 0], yb) if n_classes <= 2 else criterion(logits, yb)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t, temperature=0.1)
            val_loss = (
                float(criterion(val_logits[:, 0], y_val_t).item())
                if n_classes <= 2
                else float(criterion(val_logits, y_val_t).item())
            )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def run_cae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    *,
    n_classes: int = 2,
    seed: int = 0xCAFE,
    select_epochs_k: int = 300,  # Match original paper default
    select_epochs_2k: int = 300,  # Match original paper default
    patience: int = 30,  # Increased from 12 for longer training
    lr: float = 1e-4,  # Match original paper learning rate
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]
    k1 = max(1, min(k, n_features))
    k2 = max(1, min(2 * k, n_features))

    model_k = _train_single_selector(
        X_train,
        y_train,
        k=k1,
        n_classes=n_classes,
        seed=seed,
        max_epochs=select_epochs_k,
        patience=patience,
        lr=lr,
        device=device,
    )
    scores = model_k.feature_scores().cpu().numpy()

    if k2 == k1:
        model_2k = model_k
        scores2 = scores.copy()
    else:
        model_2k = _train_single_selector(
            X_train,
            y_train,
            k=k2,
            n_classes=n_classes,
            seed=seed + 1,
            max_epochs=select_epochs_2k,
            patience=patience,  # Use same patience for 2k model
            lr=lr,
            device=device,
        )
        scores2 = model_2k.feature_scores().cpu().numpy()

    y_train_hat = _predict_proba(model_2k, X_train, device)
    y_hat = _predict_proba(model_2k, X_test, device)
    return y_train_hat, y_hat, scores, scores2
