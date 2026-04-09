from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def _ensure_vendor_path() -> None:
    vendor_dir = Path(__file__).resolve().parents[2] / "vendor_pkgs"
    if not vendor_dir.exists():
        raise ImportError(f"Missing local vendor directory: {vendor_dir}")
    vendor_str = str(vendor_dir)
    if vendor_str not in sys.path:
        sys.path.append(vendor_str)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _to_probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 1:
        return 1.0 / (1.0 + np.exp(-logits))
    if logits.ndim == 2 and logits.shape[1] == 1:
        return 1.0 / (1.0 + np.exp(-logits[:, 0]))
    probs = _softmax(logits)
    if probs.shape[1] == 2:
        return probs[:, 1]
    return probs


def _fit_selector(
    n_features_to_select: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    max_epochs: int,
) -> object:
    _ensure_vendor_path()
    torch.set_float32_matmul_precision("high")
    from e2efs.models import E2EFS

    selector = E2EFS(
        n_features_to_select=n_features_to_select,
        network="three_layer_nn",
        precision="32",
    )
    selector.fit(
        X_train.astype(np.float32),
        y_train.astype(np.int64),
        validation_data=(X_val.astype(np.float32), y_val.astype(np.int64)),
        batch_size=batch_size,
        max_epochs=max_epochs,
        verbose=False,
    )
    return selector


def run_e2efs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    *,
    batch_size: int = 64,
    max_epochs: int = 50,
    val_split: float = 0.2,
    seed: int = 0xCAFE,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_features = X_train.shape[1]
    n_selected = max(1, min(k, n_features))
    n_selected_2k = max(1, min(2 * k, n_features))

    try:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_split,
            random_state=seed,
            stratify=y_train,
        )
    except ValueError:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_split,
            random_state=seed,
        )

    selector = _fit_selector(
        n_selected,
        X_fit,
        y_fit,
        X_val,
        y_val,
        batch_size=batch_size,
        max_epochs=max_epochs,
    )
    scores = np.asarray(selector.get_mask(), dtype=float).reshape(-1)

    if n_selected_2k == n_selected:
        scores2 = scores.copy()
        # Predict from the same model when k == 2k
        y_train_hat = _to_probabilities(
            selector.predict(
                X_train.astype(np.float32), batch_size=batch_size, verbose=False
            )
        )
        y_hat = _to_probabilities(
            selector.predict(
                X_test.astype(np.float32), batch_size=batch_size, verbose=False
            )
        )
    else:
        selector_2k = _fit_selector(
            n_selected_2k,
            X_fit,
            y_fit,
            X_val,
            y_val,
            batch_size=batch_size,
            max_epochs=max_epochs,
        )
        scores2 = np.asarray(selector_2k.get_mask(), dtype=float).reshape(-1)
        # Predict from the 2k model for consistency with FSNet and CAE
        y_train_hat = _to_probabilities(
            selector_2k.predict(
                X_train.astype(np.float32), batch_size=batch_size, verbose=False
            )
        )
        y_hat = _to_probabilities(
            selector_2k.predict(
                X_test.astype(np.float32), batch_size=batch_size, verbose=False
            )
        )

    return y_train_hat, y_hat, scores, scores2
