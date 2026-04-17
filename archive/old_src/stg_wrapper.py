# -*- coding: utf-8 -*-
"""
Thin wrapper around the official ``stg`` package (Yamada et al., 2020).

This keeps the local return signature aligned with the rest of the codebase:
``(y_train_hat, y_hat, scores, scores2)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from stg import STG


def run_stg_fs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    *,
    hidden_dims: Sequence[int] = (32, 32),
    sigma: float = 0.5,
    lam: float = 0.1,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 300,
    weight_decay: float = 1e-3,
    optimizer: str = "Adam",
    random_state: int = 0xCAFE,
    device: str | None = None,
    valid_X: np.ndarray | None = None,
    valid_y: np.ndarray | None = None,
    early_stop=None,
    shuffle: bool = False,
    print_interval: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    """Run STG feature selection and return predictions plus gate scores."""

    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = max(int(n_classes), 2)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train_int = np.asarray(y_train, dtype=np.int64)
    valid_y_int = None if valid_y is None else np.asarray(valid_y, dtype=np.int64)

    model = STG(
        device=run_device,
        input_dim=X_train.shape[1],
        output_dim=output_dim,
        hidden_dims=list(hidden_dims),
        activation="relu",
        sigma=sigma,
        lam=lam,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        feature_selection=True,
        weight_decay=weight_decay,
        task_type="classification",
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train_int,
        nr_epochs=epochs,
        valid_X=valid_X,
        valid_y=valid_y_int,
        early_stop=early_stop,
        print_interval=epochs + 1 if print_interval is None else print_interval,
        shuffle=shuffle,
    )

    scores = np.asarray(model.get_gates(mode="prob"), dtype=np.float32).reshape(-1)
    scores2 = scores.copy()

    # STG predict can hit device mismatch when the internal model stays on GPU.
    model._model.to("cpu")
    model.device = "cpu"
    y_train_pred = model.predict(X_train, verbose=False)
    y_hat_pred = model.predict(X_test, verbose=False)

    if n_classes <= 2:
        if y_train_pred.ndim == 2:
            y_train_hat = y_train_pred[:, 1]
            y_hat = y_hat_pred[:, 1]
        else:
            y_train_hat = np.asarray(y_train_pred).reshape(-1)
            y_hat = np.asarray(y_hat_pred).reshape(-1)
    else:
        if y_hat_pred.ndim == 2 and y_hat_pred.shape[1] == n_classes:
            y_train_hat = y_train_pred
            y_hat = y_hat_pred
        else:
            y_train_hat = None
            y_hat = None

    return y_train_hat, y_hat, scores, scores2
