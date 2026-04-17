"""Mentor-directed experiments along three axes:

1. gating: linear/unbounded gate vs sigmoid-bounded gate
2. backbone: MLP vs gated token transformer (+ optional masked pretraining)
3. training: select-then-MLP vs expand-then-select-then-MLP
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.admm_input_group_wrapper import (  # noqa: E402
    GatedFeatureSelectionMLP,
    _extract_feature_importance,
    _predict_proba,
    _Scaler,
    _train_input_group,
)
from src.data import generate_dataset  # noqa: E402
from src.mentor_models import (  # noqa: E402
    ExpandedFeatureSelectionMLP,
    GatedTokenTransformerFS,
)

RESULTS_DIR = os.path.join(ROOT, "results", "mentor_axes")
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class ModelSpec:
    name: str
    axis: str
    desc: str
    factory: Callable[[int, int], torch.nn.Module]
    pretrain: bool = False


AXIS_TASKS = {
    "gating": [("xor", 2, 128), ("ring", 2, 128), ("ring+xor", 4, 256)],
    "backbone": [("xor", 2, 128), ("ring+xor", 4, 256)],
    "training": [("xor", 2, 256), ("ring", 2, 128), ("ring+xor", 4, 256)],
}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _binary_auc(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    y_hat = np.asarray(y_hat).reshape(-1)
    return float(roc_auc_score(y_true, y_hat))


def _pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def pretrain_masked_transformer(
    model: GatedTokenTransformerFS,
    X_train: np.ndarray,
    *,
    device: str,
    epochs: int = 12,
    batch_size: int = 64,
    mask_prob: float = 0.15,
    lr: float = 1e-3,
) -> dict:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    tensor_x = torch.tensor(X_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses: list[float] = []

    for _ in range(epochs):
        epoch_losses = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            mask = torch.rand_like(batch_x) < mask_prob
            batch_x_masked = batch_x.clone()
            batch_x_masked[mask] = 0.0
            recon = model.reconstruct_masked(batch_x_masked)
            if mask.any():
                loss = ((recon - batch_x) ** 2)[mask].mean()
            else:
                loss = ((recon - batch_x) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    return {
        "epochs": epochs,
        "mask_prob": mask_prob,
        "mean_recon_loss": float(np.mean(losses)),
        "final_recon_loss": float(losses[-1]),
    }


def make_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="linear_gate_mlp",
            axis="gating",
            desc="Unbounded linear gate + baseline MLP",
            factory=lambda m, n_classes: GatedFeatureSelectionMLP(
                input_size=m,
                n_classes=n_classes,
                latent_size=32,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=False,
                activation="mish",
                dropout=0.043,
            ),
        ),
        ModelSpec(
            name="sigmoid_gate_mlp",
            axis="gating",
            desc="Sigmoid-bounded gate + baseline MLP",
            factory=lambda m, n_classes: GatedFeatureSelectionMLP(
                input_size=m,
                n_classes=n_classes,
                latent_size=32,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=True,
                activation="mish",
                dropout=0.043,
            ),
        ),
        ModelSpec(
            name="gated_mlp",
            axis="backbone",
            desc="Baseline gated MLP backbone",
            factory=lambda m, n_classes: GatedFeatureSelectionMLP(
                input_size=m,
                n_classes=n_classes,
                latent_size=32,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=False,
                activation="mish",
                dropout=0.043,
            ),
        ),
        ModelSpec(
            name="gated_token_transformer",
            axis="backbone",
            desc="Feature-token transformer with ADMM gate",
            factory=lambda m, n_classes: GatedTokenTransformerFS(
                input_size=m,
                n_classes=n_classes,
                d_model=16,
                n_heads=4,
                n_layers=1,
                ff_dim=64,
                feat_drop=0.6,
                bounded_gate=False,
                dropout=0.1,
            ),
        ),
        ModelSpec(
            name="gated_token_transformer_pretrained",
            axis="backbone",
            desc="Feature-token transformer with masked pretraining + ADMM gate",
            factory=lambda m, n_classes: GatedTokenTransformerFS(
                input_size=m,
                n_classes=n_classes,
                d_model=16,
                n_heads=4,
                n_layers=1,
                ff_dim=64,
                feat_drop=0.6,
                bounded_gate=False,
                dropout=0.1,
            ),
            pretrain=True,
        ),
        ModelSpec(
            name="select_then_mlp",
            axis="training",
            desc="Current order: gate raw features, then MLP",
            factory=lambda m, n_classes: GatedFeatureSelectionMLP(
                input_size=m,
                n_classes=n_classes,
                latent_size=32,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=False,
                activation="mish",
                dropout=0.043,
            ),
        ),
        ModelSpec(
            name="expand_then_select_then_mlp",
            axis="training",
            desc="Expand features into d channels, then gate, then MLP",
            factory=lambda m, n_classes: ExpandedFeatureSelectionMLP(
                input_size=m,
                n_classes=n_classes,
                expand_dim=8,
                latent_size=64,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=False,
                activation="mish",
                dropout=0.043,
            ),
        ),
    ]


def evaluate_spec(
    spec: ModelSpec,
    ds_name: str,
    k_true: int,
    m: int,
    *,
    n_folds: int,
    device: str,
    epochs: int,
    warmup_epochs: int,
) -> dict:
    _set_seed(0)
    X, _, y = generate_dataset(ds_name, 1000, m)
    X = 2.0 * X - 1.0

    fold_metrics = []
    splitter = KFold(n_splits=n_folds)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        perm_rng = np.random.RandomState(1000 + fold_idx)
        perm = perm_rng.permutation(m)
        X_train = X_train[:, perm]
        X_test = X_test[:, perm]
        correct = set(np.where(perm < k_true)[0].tolist())

        scaler = _Scaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        _set_seed(2000 + fold_idx)
        model = spec.factory(m, 2)

        pretrain_info = None
        if spec.pretrain and isinstance(model, GatedTokenTransformerFS):
            pretrain_info = pretrain_masked_transformer(
                model,
                X_train_s,
                device=device,
                epochs=12,
                batch_size=64,
                mask_prob=0.15,
            )

        _train_input_group(
            model,
            X_train_s,
            y_train,
            n_classes=2,
            lr=0.005,
            C=0.05,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            batch_size=64,
            rho_init=20.0
            if m < 64
            else (50.0 if m < 256 else (100.0 if m < 512 else 200.0)),
            device=device,
            use_ratio_norm=True,
            use_admm=True,
            n_features=m,
            optimizer_type="adam",
            use_early_stopping=True,
            patience=24,
            val_split=0.2,
        )

        scores = _extract_feature_importance(model, X_train_s)
        ranked = np.argsort(np.abs(scores))
        best_k = sum(i in correct for i in ranked[-k_true:]) / k_true
        y_hat = _predict_proba(model, X_test_s, n_classes=2)
        auc = _binary_auc(y_test, y_hat)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "best_k": float(best_k),
                "auc": float(auc),
                "pretrain": pretrain_info,
            }
        )

    return {
        "mean_best_k": float(np.mean([f["best_k"] for f in fold_metrics])),
        "std_best_k": float(np.std([f["best_k"] for f in fold_metrics])),
        "mean_auc": float(np.mean([f["auc"] for f in fold_metrics])),
        "std_auc": float(np.std([f["auc"] for f in fold_metrics])),
        "per_fold": fold_metrics,
        "desc": spec.desc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--axis", choices=["gating", "backbone", "training", "all"], default="all"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--warmup-epochs", type=int, default=48)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = _pick_device(args.device)
    specs = make_model_specs()
    wanted_axes = (
        {"gating", "backbone", "training"} if args.axis == "all" else {args.axis}
    )

    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "axis": sorted(wanted_axes),
            "device": device,
            "folds": args.folds,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
        },
        "axes": {},
    }

    for axis in sorted(wanted_axes):
        axis_results = {"tasks": {}, "models": []}
        axis_specs = [s for s in specs if s.axis == axis]
        axis_results["models"] = [
            {"name": s.name, "desc": s.desc, "pretrain": s.pretrain} for s in axis_specs
        ]
        axis_tasks = AXIS_TASKS[axis]
        if args.task_limit > 0:
            axis_tasks = axis_tasks[: args.task_limit]
        for ds_name, k_true, m in axis_tasks:
            task_key = f"{ds_name}_m{m}"
            axis_results["tasks"][task_key] = {}
            print(f"\n=== axis={axis} task={task_key} ===")
            for spec in axis_specs:
                print(f"Running {spec.name} ...")
                res = evaluate_spec(
                    spec,
                    ds_name,
                    k_true,
                    m,
                    n_folds=args.folds,
                    device=device,
                    epochs=args.epochs,
                    warmup_epochs=args.warmup_epochs,
                )
                axis_results["tasks"][task_key][spec.name] = res
                print(
                    f"  best-k={res['mean_best_k']:.3f} +/- {res['std_best_k']:.3f} | "
                    f"auc={res['mean_auc']:.3f} +/- {res['std_auc']:.3f}"
                )
        results["axes"][axis] = axis_results

    out_name = args.output or f"mentor_axes_{results['metadata']['timestamp']}.json"
    out_path = os.path.join(RESULTS_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
