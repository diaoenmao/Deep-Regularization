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


def run_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 0xCAFE,
    max_epochs: int = 100,
    patience: int = 20,
):
    _ensure_vendor_path()
    from pytorch_tabnet.tab_model import TabNetClassifier

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

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 2e-2},
        mask_type="sparsemax",
        device_name=device_name,
        seed=seed,
        verbose=0,
    )
    clf.fit(
        X_fit.astype(np.float32),
        y_fit,
        eval_set=[(X_val.astype(np.float32), y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    y_train_hat = clf.predict_proba(X_train.astype(np.float32))
    y_hat = clf.predict_proba(X_test.astype(np.float32))
    scores = np.asarray(clf.feature_importances_, dtype=float).reshape(-1)
    return y_train_hat[:, 1], y_hat[:, 1], scores, scores.copy()
