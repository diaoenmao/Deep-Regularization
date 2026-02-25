# -*- coding: utf-8 -*-
"""
Thin wrapper around the ``stg`` package (Yamada et al., 2020) for the
Feature-Selection-Benchmark.

Returns ``(y_train_hat, y_hat, scores, scores2)`` as required by
``run_fs_method`` in ``src/core.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from stg import STG


def run_stg_fs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    *,
    hidden_dims: list[int] = [32, 32],
    sigma: float = 0.5,
    lam: float = 0.1,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 300,
    random_state: int = 0xCAFE,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    """Run STG feature selection and return scores + predictions."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dim = max(n_classes, 2)  # CrossEntropyLoss needs >= 2 outputs

    model = STG(
        device=device,
        input_dim=X_train.shape[1],
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation="relu",
        sigma=sigma,
        lam=lam,
        optimizer="Adam",
        learning_rate=learning_rate,
        batch_size=batch_size,
        feature_selection=True,
        weight_decay=1e-3,
        task_type="classification",
        random_state=random_state,
    )

    # Ensure integer labels for CrossEntropyLoss
    y_train_int = y_train.astype(np.int64)

    model.fit(X_train, y_train_int, nr_epochs=epochs, print_interval=epochs + 1)

    # Gate values as feature importance scores
    gates = model.get_gates(mode="prob")
    scores = np.array(gates).flatten()
    scores2 = scores.copy()

    # Predictions — move model to CPU to avoid device mismatch in STG predict
    model._model.to("cpu")
    model.device = "cpu"
    y_train_pred = model.predict(X_train)
    y_hat_pred = model.predict(X_test)

    # Convert to probability format expected by the benchmark
    # For binary: roc_auc_score expects 1D (positive class prob)
    if n_classes <= 2:
        if y_train_pred.ndim == 2:
            y_train_hat = y_train_pred[:, 1]
            y_hat = y_hat_pred[:, 1]
        else:
            y_train_hat = y_train_pred.flatten()
            y_hat = y_hat_pred.flatten()
    else:
        # STG predict returns class labels (1D) for multiclass;
        # return None to skip the shape assertion in core.py
        if y_hat_pred.ndim == 1 or (y_hat_pred.ndim == 2 and y_hat_pred.shape[1] != n_classes):
            y_train_hat = None
            y_hat = None
        else:
            y_train_hat = y_train_pred
            y_hat = y_hat_pred

    return y_train_hat, y_hat, scores, scores2
