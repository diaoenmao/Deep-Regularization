from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

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
from src.data import generate_dataset  # noqa: E402
from src.mentor_models import ExpandedFeatureSelectionMLP  # noqa: E402

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
    desc: str
    factory: Callable[[int], torch.nn.Module]


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


def _make_methods() -> dict[str, MethodSpec]:
    return {
        "select_then_mlp": MethodSpec(
            name="select_then_mlp",
            desc="Baseline order: select on raw features, then MLP",
            factory=lambda m: GatedFeatureSelectionMLP(
                input_size=m,
                n_classes=2,
                latent_size=32,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=False,
                activation="mish",
                dropout=0.043,
            ),
        ),
        "expand4_then_select_then_mlp": MethodSpec(
            name="expand4_then_select_then_mlp",
            desc="Expand to 4 channels per feature, then gate, then MLP",
            factory=lambda m: ExpandedFeatureSelectionMLP(
                input_size=m,
                n_classes=2,
                expand_dim=4,
                latent_size=64,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=False,
                activation="mish",
                dropout=0.043,
            ),
        ),
        "expand8_then_select_then_mlp": MethodSpec(
            name="expand8_then_select_then_mlp",
            desc="Expand to 8 channels per feature, then gate, then MLP",
            factory=lambda m: ExpandedFeatureSelectionMLP(
                input_size=m,
                n_classes=2,
                expand_dim=8,
                latent_size=64,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=False,
                activation="mish",
                dropout=0.043,
            ),
        ),
        "expand16_then_select_then_mlp": MethodSpec(
            name="expand16_then_select_then_mlp",
            desc="Expand to 16 channels per feature, then gate, then MLP",
            factory=lambda m: ExpandedFeatureSelectionMLP(
                input_size=m,
                n_classes=2,
                expand_dim=16,
                latent_size=96,
                n_hidden_layers=2,
                feat_drop=0.6,
                bounded_gate=False,
                activation="mish",
                dropout=0.043,
            ),
        ),
    }


def _init_payload(metadata: dict, methods: dict[str, MethodSpec]) -> dict:
    return {
        "metadata": metadata,
        "methods": {name: {"desc": spec.desc} for name, spec in methods.items()},
        "results": {},
    }


def _load_or_init(path: str, metadata: dict, methods: dict[str, MethodSpec]) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("results", {})
        data.setdefault("methods", {})
        for name, spec in methods.items():
            data["methods"].setdefault(name, {"desc": spec.desc})
        return data
    return _init_payload(metadata, methods)


def _save(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--warmup-epochs", type=int, default=60)
    parser.add_argument("--dataset-filter", nargs="+", default=None)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--task-keys", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = _pick_device(args.device)
    all_methods = _make_methods()
    selected = args.methods or list(all_methods.keys())
    methods = {name: all_methods[name] for name in selected}

    dataset_filter = set(args.dataset_filter or [])
    task_keys_filter = set(args.task_keys or [])
    tasks: list[tuple[str, int, int]] = []
    for ds_name, k_true, dims in DATASETS_CONFIG:
        if dataset_filter and ds_name not in dataset_filter:
            continue
        for m in dims:
            task_key = f"{ds_name}_m{m}"
            if task_keys_filter and task_key not in task_keys_filter:
                continue
            tasks.append((ds_name, k_true, m))
    if args.task_limit > 0:
        tasks = tasks[: args.task_limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.output or f"training_order_synthetic_full_{stamp}.json"
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
        "task_keys": sorted(task_keys_filter),
        "task_limit": args.task_limit,
        "protocol": "training-order synthetic grid with 6-fold CV, feature permutation, best-k/best-2k/AUC/AUPRC",
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
        X, _, y = generate_dataset(ds_name, N_SAMPLES, m)
        X = 2.0 * X - 1.0
        splits = list(KFold(n_splits=N_FOLDS).split(X))

        for method_name, spec in methods.items():
            if method_name in payload["results"][task_key]:
                done += 1
                continue

            print(f"\n[{done + 1}/{total}] task={task_key} method={method_name}")
            t0 = time.time()
            fold_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                perm_rng = np.random.RandomState(1000 * task_idx + fold_idx)
                perm = perm_rng.permutation(m)
                X_train = X_train[:, perm]
                X_test = X_test[:, perm]
                correct = set(np.where(perm < k_true)[0].tolist())

                scaler = _Scaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                fold_seed = 10000 * task_idx + fold_idx
                _set_seed(fold_seed)
                model = spec.factory(m)
                _train_input_group(
                    model,
                    X_train_s,
                    y_train,
                    n_classes=2,
                    lr=0.005,
                    C=0.05,
                    epochs=args.epochs,
                    warmup_epochs=args.warmup_epochs,
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
                )

                scores = _extract_feature_importance(model, X_train_s)
                ranked = np.argsort(np.abs(scores))
                best_k = float(sum(i in correct for i in ranked[-k_true:]) / k_true)
                ranked2 = np.argsort(np.abs(scores))
                best_2k = float(
                    sum(i in correct for i in ranked2[-(2 * k_true) :]) / k_true
                )
                y_hat = _predict_proba(model, X_test_s, n_classes=2)
                test_scores = _binary_scores(y_hat)
                auc = float(roc_auc_score(y_test, test_scores))
                auprc = float(average_precision_score(y_test, test_scores))
                fold_rows.append(
                    {
                        "fold": fold_idx,
                        "best_k": best_k,
                        "best_2k": best_2k,
                        "auc": auc,
                        "auprc": auprc,
                    }
                )
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
