#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Controlled synthetic benchmark for neural feature-selection methods.

This script is intentionally separated from Feature-Selection-Benchmark/ so we
can run same-backbone experiments without changing benchmark source code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
import torch
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.admm_input_group_wrapper import run_admm_input_group
from src.data import generate_dataset
from src.fsnet import FSNet
from src.nn_wrapper import Model, NNwrapper
from src.stg_wrapper import run_stg_fs

SEED = 0
N_SAMPLES = 1000
N_FOLDS = 6

MATCHED_BACKBONE = {
    "latent_size": 32,
    "n_hidden_layers": 2,
    "gaussian_noise": 0.0,
    "dropout": 0.0,
    "activation": "mish",
}

MATCHED_NN_FIT = {
    "learning_rate": 0.0017601777068292975,
    "epochs": 416,
    "batch_size": 56,
    "weight_decay": 0.00048519293899787247,
    "val": 0.2,
    "early_stopping_patience": 66,
    "optimizer": "adagrad",
    "sam_type": "no-sam",
}

MATCHED_SADMM_HP = {
    "latent_size": 32,
    "n_hidden_layers": 2,
    "gaussian_noise": 0.0,
    "dropout": 0.0,
    "activation": "mish",
    "epochs": 416,
    "warmup_epochs": 100,
    "batch_size": 56,
    "optimizer_type": "adagrad",
    "use_early_stopping": True,
    "patience": 66,
    "val_split": 0.2,
    "feat_drop": 0.6,
}

FSNET_FIT = {
    "n_epochs": 416,
    "batch_size": 56,
    "_lambda": 10.0,
    "weight_decay": 1e-6,
}

LASONET_KWARGS = {
    "hidden_dims": (32, 32),
    "n_iters": (30, 30),
    "batch_size": 64,
    "dropout": 0.0,
    "patience": 10,
    "lambda_start": 3.2768,
    "tol": 0.9999,
    "verbose": False,
}

QUICK_CONFIG = {
    "xor": (2, [8, 128]),
    "ring": (2, [32]),
    "ring+xor": (4, [16]),
}

FULL_CONFIG = {
    "xor": (2, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
    "ring": (2, [8, 16, 32, 64, 128, 256, 512, 1024]),
    "ring+xor": (4, [4, 8, 16, 32, 64, 128, 256, 512]),
}

METHOD_SPECS = {
    "sadmm_fs": {
        "label": "SADMM-FS",
        "match_level": "method_specific",
        "reason": "Shared 2x32 predictor and Adagrad family, but ADMM gate updates are method-specific.",
    },
    "cancelout_sigmoid": {
        "label": "CancelOut-Sigmoid",
        "match_level": "full_match",
        "reason": "Shared 2x32 predictor and shared NNwrapper training config.",
    },
    "cancelout_softmax": {
        "label": "CancelOut-Softmax",
        "match_level": "full_match",
        "reason": "Shared 2x32 predictor and shared NNwrapper training config.",
    },
    "deeppink": {
        "label": "DeepPINK",
        "match_level": "full_match",
        "reason": "Shared 2x32 predictor and shared NNwrapper training config; method-specific knockoff front-end retained.",
    },
    "fsnet": {
        "label": "FSNet",
        "match_level": "backbone_only",
        "reason": "Predictor width/depth matched, but FSNet uses its own selector and Adam-based training loop.",
    },
    "lassonet": {
        "label": "LassoNet",
        "match_level": "backbone_only",
        "reason": "Hidden dims matched to 2x32, but optimization path is LassoNet-specific.",
    },
    "stg": {
        "label": "STG",
        "match_level": "backbone_only",
        "reason": "Hidden dims matched to 2x32, but STG keeps its stochastic-gate objective and Adam training path.",
    },
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_best_k(scores: np.ndarray, correct: set[int], k: int) -> float:
    ranked = np.argsort(np.abs(scores))
    return float(sum(i in correct for i in ranked[-k:]) / k)


def prepare_fold(
    dataset_name: str,
    k: int,
    n_features: int,
    fold_idx: int,
) -> dict:
    set_seed(SEED + n_features)
    X, X_tilde, y = generate_dataset(dataset_name, N_SAMPLES, n_features)
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    splits = list(KFold(n_splits=N_FOLDS).split(X))
    train_idx, test_idx = splits[fold_idx]

    X_train, X_test = X[train_idx], X[test_idx]
    X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
    y_train = y[train_idx]

    rng = np.random.RandomState(SEED + n_features * 10 + fold_idx)
    feature_order = np.arange(n_features)
    rng.shuffle(feature_order)

    X_train = X_train[:, feature_order]
    X_test = X_test[:, feature_order]
    X_tilde_train = X_tilde_train[:, feature_order]
    X_tilde_test = X_tilde_test[:, feature_order]
    correct = set(np.where(feature_order < k)[0].tolist())

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_tilde_train": X_tilde_train,
        "X_tilde_test": X_tilde_test,
        "y_train": y_train,
        "correct": correct,
        "n_features": n_features,
        "k": k,
    }


def run_sadmm_fs(fold: dict) -> np.ndarray:
    _, _, scores, _ = run_admm_input_group(
        fold["X_train"],
        fold["y_train"],
        fold["X_test"],
        n_classes=2,
        hp_overrides=MATCHED_SADMM_HP,
        seed=SEED,
    )
    return scores


def run_cancelout(fold: dict, activation: str) -> np.ndarray:
    arch = f"cancelout-{activation}"
    wrapper = NNwrapper.create(
        "synthetic",
        fold["n_features"],
        2,
        arch=arch,
        model_kwargs=MATCHED_BACKBONE,
    )
    wrapper.fit(fold["X_train"], fold["y_train"], **MATCHED_NN_FIT)
    return wrapper.model.cancel_out.get_weights().detach().cpu().numpy()


def run_deeppink(fold: dict) -> np.ndarray:
    X_aug_train = np.empty((fold["X_train"].shape[0], fold["X_train"].shape[1], 2))
    X_aug_train[:, :, 0] = fold["X_train"]
    X_aug_train[:, :, 1] = fold["X_tilde_train"]

    wrapper = NNwrapper.create(
        "synthetic",
        fold["n_features"],
        2,
        arch="deeppink",
        model_kwargs=MATCHED_BACKBONE,
    )
    wrapper.fit(X_aug_train, fold["y_train"], **MATCHED_NN_FIT)
    scores = wrapper.model.get_weights()
    scores = scores - scores.min()
    return scores


def run_fsnet(fold: dict) -> np.ndarray:
    n_selected = min(2 * fold["k"], fold["n_features"])
    predictor = Model(n_selected, 2, **MATCHED_BACKBONE)
    selector = FSNet(predictor, fold["n_features"], 30, n_selected, 2)
    selector.fit(fold["X_train"], fold["y_train"], **FSNET_FIT)
    return selector.get_feature_importances()


def run_lassonet(fold: dict) -> np.ndarray:
    import lassonet

    model = lassonet.LassoNetClassifier(**LASONET_KWARGS)
    model.path(fold["X_train"], fold["y_train"], return_state_dicts=True)
    return model.feature_importances_.numpy()


def run_stg(fold: dict) -> np.ndarray:
    _, _, scores, _ = run_stg_fs(
        fold["X_train"],
        fold["y_train"],
        fold["X_test"],
        n_classes=2,
        hidden_dims=(MATCHED_BACKBONE["latent_size"],)
        * MATCHED_BACKBONE["n_hidden_layers"],
        learning_rate=1e-3,
        batch_size=MATCHED_NN_FIT["batch_size"],
        epochs=MATCHED_NN_FIT["epochs"],
        random_state=SEED,
    )
    return scores


RUNNERS: Dict[str, Callable[[dict], np.ndarray]] = {
    "sadmm_fs": run_sadmm_fs,
    "cancelout_sigmoid": lambda fold: run_cancelout(fold, "sigmoid"),
    "cancelout_softmax": lambda fold: run_cancelout(fold, "softmax"),
    "deeppink": run_deeppink,
    "fsnet": run_fsnet,
    "lassonet": run_lassonet,
    "stg": run_stg,
}


def run_method(method_name: str, fold: dict) -> dict:
    t0 = time.time()
    try:
        scores = RUNNERS[method_name](fold)
        return {
            "status": "ok",
            "best_k": compute_best_k(scores, fold["correct"], fold["k"]),
            "runtime_sec": time.time() - t0,
        }
    except ImportError as exc:
        return {
            "status": "unavailable",
            "error": f"import error: {exc}",
        }
    except Exception as exc:  # pragma: no cover - experiment runner
        return {
            "status": "failed",
            "error": str(exc),
        }


def default_methods() -> List[str]:
    return [
        "sadmm_fs",
        "cancelout_sigmoid",
        "cancelout_softmax",
        "deeppink",
        "fsnet",
        "lassonet",
        "stg",
    ]


def summarise(results: dict) -> dict:
    summary = {}
    for method_name, method_results in results["results"].items():
        dataset_scores = []
        dataset_summary = {}
        for dataset_name, dims in method_results.items():
            dim_scores = []
            dim_runtimes = []
            for dim_key, payload in dims.items():
                fold_scores = [
                    fold["best_k"]
                    for fold in payload["folds"]
                    if fold["status"] == "ok"
                ]
                fold_runtimes = [
                    fold["runtime_sec"]
                    for fold in payload["folds"]
                    if fold["status"] == "ok"
                ]
                dim_scores.extend(fold_scores)
                dim_runtimes.extend(fold_runtimes)
                payload["mean_best_k"] = (
                    float(np.mean(fold_scores)) if fold_scores else None
                )
                payload["std_best_k"] = (
                    float(np.std(fold_scores)) if fold_scores else None
                )
                payload["mean_runtime_sec"] = (
                    float(np.mean(fold_runtimes)) if fold_runtimes else None
                )

            dataset_summary[dataset_name] = {
                "mean_best_k": float(np.mean(dim_scores)) if dim_scores else None,
                "mean_runtime_sec": float(np.mean(dim_runtimes))
                if dim_runtimes
                else None,
            }
            if dim_scores:
                dataset_scores.extend(dim_scores)

        summary[method_name] = {
            "overall_mean_best_k": float(np.mean(dataset_scores))
            if dataset_scores
            else None,
            "datasets": dataset_summary,
            "match_level": METHOD_SPECS[method_name]["match_level"],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a reduced config.")
    parser.add_argument("--methods", nargs="+", default=default_methods())
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--dim-limit", type=int, default=None)
    args = parser.parse_args()

    config = QUICK_CONFIG if args.quick else FULL_CONFIG
    if args.datasets:
        config = {name: config[name] for name in args.datasets}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "results", "matched_neural")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"matched_neural_synthetic_{timestamp}.json")

    results = {
        "metadata": {
            "timestamp": timestamp,
            "quick_mode": args.quick,
            "n_samples": N_SAMPLES,
            "n_folds": N_FOLDS,
            "fold_limit": args.fold_limit,
            "dim_limit": args.dim_limit,
            "matched_backbone": MATCHED_BACKBONE,
            "matched_nn_fit": MATCHED_NN_FIT,
            "matched_sadmm_hp": MATCHED_SADMM_HP,
            "fsnet_fit": FSNET_FIT,
            "lassonet_kwargs": LASONET_KWARGS,
        },
        "method_specs": {
            name: METHOD_SPECS[name]
            for name in list(args.methods) + ["stg"]
            if name in METHOD_SPECS
        },
        "results": {name: {} for name in args.methods},
    }

    for method_name in args.methods:
        if method_name not in RUNNERS:
            continue
        for dataset_name, (k, dimensions) in config.items():
            results["results"][method_name][dataset_name] = {}
            dim_values = dimensions[: args.dim_limit] if args.dim_limit else dimensions
            fold_count = args.fold_limit if args.fold_limit else N_FOLDS
            for n_features in dim_values:
                fold_payloads = []
                for fold_idx in range(fold_count):
                    fold = prepare_fold(dataset_name, k, n_features, fold_idx)
                    fold_payload = run_method(method_name, fold)
                    fold_payload["fold_idx"] = fold_idx
                    fold_payloads.append(fold_payload)
                results["results"][method_name][dataset_name][str(n_features)] = {
                    "k": k,
                    "folds": fold_payloads,
                }

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

    results["summary"] = summarise(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
