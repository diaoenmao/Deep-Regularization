from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.admm_input_group_wrapper import (  # noqa: E402
    GatedFeatureSelectionMLP,
    _extract_feature_importance,
    _predict_proba,
    _Scaler,
    _train_input_group,
)
from src.cae_wrapper import run_cae  # noqa: E402
from src.core import run_fs_method  # noqa: E402
from src.data import generate_dataset  # noqa: E402
from src.e2efs_wrapper import run_e2efs  # noqa: E402
from src.mentor_models import GatedTokenTransformerFS  # noqa: E402
from src.tabnet_wrapper import run_tabnet  # noqa: E402

N_SAMPLES = 1000
N_FOLDS = 6
RESULTS_DIR = os.path.join(ROOT, "results", "mentor_axes")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS_CONFIG = [
    ("xor", 2, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring", 2, [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor", 4, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor+sum", 6, [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
]


@dataclass
class MethodSpec:
    name: str
    family: str
    desc: str
    runner: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    requires_device: bool = False
    pretrain: bool = False


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _binary_scores(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr[:, 1]
    return arr.reshape(-1)


def _rho_for_dim(m: int) -> float:
    if m < 64:
        return 20.0
    if m < 256:
        return 50.0
    if m < 512:
        return 100.0
    return 200.0


def _pretrain_masked_transformer(
    model: GatedTokenTransformerFS,
    X_train: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    mask_prob: float,
    lr: float,
) -> dict:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    tensor_x = torch.tensor(X_train, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor_x),
        batch_size=batch_size,
        shuffle=True,
    )
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
    return {
        "epochs": epochs,
        "mask_prob": mask_prob,
        "mean_recon_loss": float(np.mean(losses)),
        "final_recon_loss": float(losses[-1]),
    }


def _run_admm_backbone(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    m: int,
    device: str,
    epochs: int,
    warmup_epochs: int,
    uniform_penalty: bool = False,
    pretrain_cfg: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
    scaler = _Scaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pretrain_info = None
    if pretrain_cfg and isinstance(model, GatedTokenTransformerFS):
        pretrain_info = _pretrain_masked_transformer(
            model,
            X_train_s,
            device=device,
            epochs=pretrain_cfg["epochs"],
            batch_size=pretrain_cfg["batch_size"],
            mask_prob=pretrain_cfg["mask_prob"],
            lr=pretrain_cfg["lr"],
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
        rho_init=_rho_for_dim(m),
        device=device,
        use_ratio_norm=True,
        use_admm=True,
        n_features=m,
        optimizer_type="adam",
        use_early_stopping=True,
        patience=24,
        val_split=0.2,
        uniform_penalty=uniform_penalty,
    )

    scores = _extract_feature_importance(model, X_train_s)
    y_train_hat = _predict_proba(model, X_train_s, n_classes=2)
    y_hat = _predict_proba(model, X_test_s, n_classes=2)
    return y_train_hat, y_hat, scores, scores, pretrain_info


def _make_methods() -> dict[str, MethodSpec]:
    def run_gated_mlp(
        X_train,
        y_train,
        X_test,
        *,
        m,
        device,
        epochs,
        warmup_epochs,
        **_,
    ):
        model = GatedFeatureSelectionMLP(
            input_size=m,
            n_classes=2,
            latent_size=32,
            n_hidden_layers=2,
            feat_drop=0.6,
            bounded_gate=False,
            activation="mish",
            dropout=0.043,
        )
        return _run_admm_backbone(
            model,
            X_train,
            y_train,
            X_test,
            m=m,
            device=device,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
        )

    def run_transformer(
        X_train,
        y_train,
        X_test,
        *,
        m,
        device,
        epochs,
        warmup_epochs,
        **_,
    ):
        model = GatedTokenTransformerFS(
            input_size=m,
            n_classes=2,
            d_model=32,
            n_heads=4,
            n_layers=2,
            ff_dim=128,
            feat_drop=0.6,
            bounded_gate=False,
            dropout=0.1,
        )
        return _run_admm_backbone(
            model,
            X_train,
            y_train,
            X_test,
            m=m,
            device=device,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            uniform_penalty=True,  # Use uniform penalty for transformer (gate-based scores)
        )

    def run_transformer_pretrained(
        X_train,
        y_train,
        X_test,
        *,
        m,
        device,
        epochs,
        warmup_epochs,
        **_,
    ):
        model = GatedTokenTransformerFS(
            input_size=m,
            n_classes=2,
            d_model=32,
            n_heads=4,
            n_layers=2,
            ff_dim=128,
            feat_drop=0.6,
            bounded_gate=False,
            dropout=0.1,
        )
        return _run_admm_backbone(
            model,
            X_train,
            y_train,
            X_test,
            m=m,
            device=device,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            uniform_penalty=True,  # Use uniform penalty for transformer (gate-based scores)
            pretrain_cfg={
                "epochs": 24,
                "batch_size": 64,
                "mask_prob": 0.15,
                "lr": 1e-3,
            },
        )

    def run_fsnet(
        X_train,
        y_train,
        X_test,
        *,
        dataset_name,
        X_tilde_train,
        X_tilde_test,
        k_true,
        **_,
    ):
        y_train_hat, y_hat, scores, scores2 = run_fs_method(
            dataset_name,
            "fsnet",
            X_train,
            X_tilde_train,
            y_train,
            X_test,
            X_tilde_test,
            k_true,
        )
        return y_train_hat, y_hat, scores, scores2, None

    def run_e2efs_method(
        X_train,
        y_train,
        X_test,
        *,
        k_true,
        **_,
    ):
        y_train_hat, y_hat, scores, scores2 = run_e2efs(
            X_train.astype(np.float32),
            y_train.astype(np.int64),
            X_test.astype(np.float32),
            k_true,
            batch_size=64,
            max_epochs=200,  # Reduced from 500 (package default), early stopping on nfeats will trigger earlier
            seed=_.get("seed", 0xCAFE),
        )
        return y_train_hat, y_hat, scores, scores2, None

    def run_cae_method(
        X_train,
        y_train,
        X_test,
        *,
        k_true,
        **_,
    ):
        y_train_hat, y_hat, scores, scores2 = run_cae(
            X_train.astype(np.float32),
            y_train,
            X_test.astype(np.float32),
            k_true,
            n_classes=2,
        )
        return y_train_hat, y_hat, scores, scores2, None

    def run_tabnet_method(
        X_train,
        y_train,
        X_test,
        *,
        seed,
        **_,
    ):
        y_train_hat, y_hat, scores, scores2 = run_tabnet(
            X_train.astype(np.float32),
            y_train,
            X_test.astype(np.float32),
            seed=seed,
            max_epochs=100,
            patience=20,
        )
        return y_train_hat, y_hat, scores, scores2, None

    return {
        "gated_mlp": MethodSpec(
            name="gated_mlp",
            family="ours_backbone",
            desc="2x32 gated MLP with ADMM",
            runner=run_gated_mlp,
            requires_device=True,
        ),
        "gated_token_transformer": MethodSpec(
            name="gated_token_transformer",
            family="ours_backbone",
            desc="2-layer token transformer with global gate + ADMM",
            runner=run_transformer,
            requires_device=True,
        ),
        "gated_token_transformer_pretrained": MethodSpec(
            name="gated_token_transformer_pretrained",
            family="ours_backbone",
            desc="Masked-pretrained token transformer with global gate + ADMM",
            runner=run_transformer_pretrained,
            requires_device=True,
            pretrain=True,
        ),
        "fsnet": MethodSpec(
            name="fsnet",
            family="tier2_baseline",
            desc="FSNet embedded baseline",
            runner=run_fsnet,
        ),
        "e2efs": MethodSpec(
            name="e2efs",
            family="tier2_baseline",
            desc="E2E-FS embedded selector",
            runner=run_e2efs_method,
        ),
        "cae": MethodSpec(
            name="cae",
            family="tier2_baseline",
            desc="Concrete selector baseline (local compatible CAE-style implementation)",
            runner=run_cae_method,
        ),
        "tabnet": MethodSpec(
            name="tabnet",
            family="tier2_baseline",
            desc="TabNet encoder-style selector using feature importances",
            runner=run_tabnet_method,
        ),
    }


def _ensure_metrics(
    y_true: np.ndarray,
    y_train_hat: np.ndarray,
    y_hat: np.ndarray,
) -> tuple[float, float]:
    test_scores = _binary_scores(y_hat)
    auc = float(roc_auc_score(y_true, test_scores))
    auprc = float(average_precision_score(y_true, test_scores))
    return auc, auprc


def _load_or_init(path: str, metadata: dict, methods: dict[str, MethodSpec]) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("results", {})
        data.setdefault("methods", {})
        for name, spec in methods.items():
            data["methods"].setdefault(
                name,
                {
                    "family": spec.family,
                    "desc": spec.desc,
                    "pretrain": spec.pretrain,
                },
            )
        return data
    return {
        "metadata": metadata,
        "methods": {
            name: {
                "family": spec.family,
                "desc": spec.desc,
                "pretrain": spec.pretrain,
            }
            for name, spec in methods.items()
        },
        "results": {},
    }


def _save(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _init_payload(metadata: dict, methods: dict[str, MethodSpec]) -> dict:
    return {
        "metadata": metadata,
        "methods": {
            name: {
                "family": spec.family,
                "desc": spec.desc,
                "pretrain": spec.pretrain,
            }
            for name, spec in methods.items()
        },
        "results": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--warmup-epochs", type=int, default=60)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--dataset-filter", nargs="+", default=None)
    parser.add_argument("--task-keys", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = _pick_device(args.device)
    all_methods = _make_methods()
    selected = args.methods or list(all_methods.keys())
    methods = {name: all_methods[name] for name in selected}

    dataset_filter = set(args.dataset_filter or [])
    task_keys_filter = args.task_keys or []
    tasks: list[tuple[str, int, int]] = []
    for ds_name, k_true, dims in DATASETS_CONFIG:
        if dataset_filter and ds_name not in dataset_filter:
            continue
        for m in dims:
            tasks.append((ds_name, k_true, m))
    if task_keys_filter:
        wanted = set(task_keys_filter)
        tasks = [
            (ds_name, k_true, m)
            for ds_name, k_true, m in tasks
            if f"{ds_name}_m{m}" in wanted
        ]
    if args.task_limit > 0:
        tasks = tasks[: args.task_limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.output or f"backbone_tier2_synthetic_full_{stamp}.json"
    out_path = os.path.join(RESULTS_DIR, out_name)

    metadata = {
        "timestamp": stamp,
        "n_samples": N_SAMPLES,
        "n_folds": N_FOLDS,
        "device": device,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "methods_requested": selected,
        "dataset_filter": sorted(dataset_filter),
        "task_keys": task_keys_filter,
        "task_limit": args.task_limit,
        "protocol": "official synthetic grid with 6-fold CV, feature permutation, best-k/best-2k/AUC/AUPRC",
        "unavailable_tier2": {},
    }
    payload = (
        _load_or_init(out_path, metadata, methods)
        if args.resume
        else _init_payload(metadata, methods)
    )

    total = len(tasks) * len(methods)
    done = 0
    for task_idx, (ds_name, k_true, m) in enumerate(tasks, start=1):
        task_key = f"{ds_name}_m{m}"
        payload["results"].setdefault(task_key, {})

        _set_seed(0)
        X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, m)
        X = 2.0 * X - 1.0
        X_tilde = 2.0 * X_tilde - 1.0
        splits = list(KFold(n_splits=N_FOLDS).split(X))

        for method_name, spec in methods.items():
            if method_name in payload["results"][task_key]:
                done += 1
                continue

            print(f"\n[{done + 1}/{total}] task={task_key} method={method_name}")
            fold_rows = []
            t0 = time.time()
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                X_train, X_test = X[train_idx], X[test_idx]
                X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                perm_rng = np.random.RandomState(1000 * task_idx + fold_idx)
                perm = perm_rng.permutation(m)
                X_train = X_train[:, perm]
                X_test = X_test[:, perm]
                X_tilde_train = X_tilde_train[:, perm]
                X_tilde_test = X_tilde_test[:, perm]
                correct = set(np.where(perm < k_true)[0].tolist())

                fold_seed = 10000 * task_idx + fold_idx
                _set_seed(fold_seed)
                y_train_hat, y_hat, scores, scores2, extra = spec.runner(
                    X_train,
                    y_train,
                    X_test,
                    dataset_name=ds_name,
                    X_tilde_train=X_tilde_train,
                    X_tilde_test=X_tilde_test,
                    k_true=k_true,
                    m=m,
                    device=device,
                    epochs=args.epochs,
                    warmup_epochs=args.warmup_epochs,
                    seed=fold_seed,
                )

                ranked = np.argsort(np.abs(scores))
                ranked2 = np.argsort(np.abs(scores2 if scores2 is not None else scores))
                best_k = float(sum(i in correct for i in ranked[-k_true:]) / k_true)
                best_2k = float(
                    sum(i in correct for i in ranked2[-(2 * k_true) :]) / k_true
                )
                auc, auprc = _ensure_metrics(y_test, y_train_hat, y_hat)
                row = {
                    "fold": fold_idx,
                    "best_k": best_k,
                    "best_2k": best_2k,
                    "auc": auc,
                    "auprc": auprc,
                }
                if extra is not None:
                    row["extra"] = extra
                fold_rows.append(row)
                print(
                    f"  fold={fold_idx} best-k={best_k:.3f} best-2k={best_2k:.3f} auc={auc:.3f}"
                )

            elapsed = time.time() - t0
            payload["results"][task_key][method_name] = {
                "mean_best_k": float(np.mean([r["best_k"] for r in fold_rows])),
                "std_best_k": float(np.std([r["best_k"] for r in fold_rows])),
                "mean_best_2k": float(np.mean([r["best_2k"] for r in fold_rows])),
                "std_best_2k": float(np.std([r["best_2k"] for r in fold_rows])),
                "mean_auc": float(np.mean([r["auc"] for r in fold_rows])),
                "std_auc": float(np.std([r["auc"] for r in fold_rows])),
                "mean_auprc": float(np.mean([r["auprc"] for r in fold_rows])),
                "std_auprc": float(np.std([r["auprc"] for r in fold_rows])),
                "elapsed_sec": float(elapsed),
                "per_fold": fold_rows,
            }
            _save(out_path, payload)
            done += 1

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
