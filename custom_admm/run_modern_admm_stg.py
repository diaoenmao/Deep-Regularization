# -*- coding: utf-8 -*-
"""
Run SADMM-FS (admm_input_group) and STG on 6 modern datasets.
Reuses data loading from real-data-benchmark.py and method dispatch from src/core.py.

Usage:
    python run_modern_admm_stg.py [--force]
"""

import argparse
import io
import json
import os
import random
import time

import mnist
import numpy as np
import pandas as pd
import PIL.Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.core import run_fs_method

SEED = 0xCAFE
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(ROOT, "results", "external-data")
os.makedirs(RESULTS_PATH, exist_ok=True)

METHODS = ["admm_input_group", "stg"]
DATASETS = ["fashion", "mnist", "coil20", "isolet", "mice", "har"]
N_SEEDS = 5


def _resolve_data_path():
    candidates = [
        os.path.join(ROOT, "data"),
        os.path.join(os.path.dirname(ROOT), "data"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _load_isolet_csv(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x9d":
        import unlzw3

        raw = unlzw3.unlzw(raw)
    return np.loadtxt(io.StringIO(raw.decode("ascii")), delimiter=",")


def load_dataset(name, seed=SEED):
    np.random.seed(seed)
    DATA_PATH = _resolve_data_path()
    if name in {"fashion", "mnist"}:
        mndata = mnist.MNIST(os.path.join(DATA_PATH, name))
        X_train, y_train = mndata.load_training()
        X_test, y_test = mndata.load_testing()
        X_train, y_train = np.asarray(X_train), np.asarray(y_train)
        X_test, y_test = np.asarray(X_test), np.asarray(y_test)
        X_train = X_train.astype(float) / 255
        X_test = X_test.astype(float) / 255
    elif name == "coil20":
        X, y = [], []
        for filename in os.listdir(os.path.join(DATA_PATH, "coil-20-proc")):
            filepath = os.path.join(DATA_PATH, "coil-20-proc", filename)
            label = int(filename.split("__")[0].replace("obj", "")) - 1
            image = np.asarray(PIL.Image.open(filepath).convert("L"))
            X.append(image.flatten())
            y.append(label)
        X = np.asarray(X).astype(float) / 255
        y = np.asarray(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
    elif name == "isolet":
        data = _load_isolet_csv(os.path.join(DATA_PATH, "isolet", "isolet1+2+3+4.data"))
        X_train = data[:, :-1]
        y_train = data[:, -1].astype(int) - 1
        data = _load_isolet_csv(os.path.join(DATA_PATH, "isolet", "isolet5.data"))
        X_test = data[:, :-1]
        y_test = data[:, -1].astype(int) - 1
    elif name == "mice":
        df = pd.read_excel(
            os.path.join(
                DATA_PATH, "mice+protein+expression", "Data_Cortex_Nuclear.xls"
            )
        )
        df.drop(columns=["MouseID", "Genotype", "Treatment", "Behavior"], inplace=True)
        y = LabelEncoder().fit_transform(df["class"].to_numpy())
        df.drop(columns=["class"], inplace=True)
        X = df.to_numpy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = np.nanmedian(X[:, j])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
    elif name == "har":
        X_train = np.loadtxt(
            os.path.join(DATA_PATH, "UCI HAR Dataset", "train", "X_train.txt")
        )
        y_train = (
            np.loadtxt(
                os.path.join(DATA_PATH, "UCI HAR Dataset", "train", "y_train.txt")
            ).astype(int)
            - 1
        )
        X_test = np.loadtxt(
            os.path.join(DATA_PATH, "UCI HAR Dataset", "test", "X_test.txt")
        )
        y_test = (
            np.loadtxt(
                os.path.join(DATA_PATH, "UCI HAR Dataset", "test", "y_test.txt")
            ).astype(int)
            - 1
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    n_classes = int(np.max(y_train)) + 1
    return X_train, y_train, X_test, y_test, n_classes


def evaluate_downstream(X_train, y_train, X_test, y_test, scores, k, seed=SEED):
    idx = np.argsort(scores)[-k:]
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    model.fit(X_train[:, idx], y_train)
    y_hat = model.predict_proba(X_test[:, idx])
    if y_hat.shape[1] == 2:
        y_hat_score = y_hat[:, 1]
        auroc = roc_auc_score(y_test, y_hat_score)
        auprc = average_precision_score(y_test, y_hat_score)
    else:
        auroc = roc_auc_score(y_test, y_hat, multi_class="ovr")
        auprc = float("nan")  # AUPRC not well-defined for multiclass
    return auroc, auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing result JSON files instead of skipping them.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Subset of methods to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Subset of datasets to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=N_SEEDS,
        help="Number of random seeds to evaluate.",
    )
    args = parser.parse_args()

    seeds = [SEED + i for i in range(args.seeds)]

    for ds in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'=' * 60}")

        X_train, y_train, X_test, y_test, n_classes = load_dataset(ds)
        n_features = X_train.shape[1]
        k = int(round(0.5 * n_features))  # default decoy_fraction=0.5
        print(
            f"  n_train={X_train.shape[0]}, n_test={X_test.shape[0]}, "
            f"n_features={n_features}, n_classes={n_classes}, k={k}"
        )

        for method in args.methods:
            out_path = os.path.join(RESULTS_PATH, f"{ds}-{method}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"  {method}: already exists, skipping")
                continue

            print(f"  Running {method}...")
            per_seed_auroc = []
            per_seed_auprc = []
            total_time = 0

            for i, s in enumerate(seeds):
                t0 = time.time()
                np.random.seed(s)
                try:
                    import torch

                    torch.manual_seed(s)
                except ImportError:
                    pass

                _, _, scores, _ = run_fs_method(
                    ds,
                    method,
                    X_train.astype(np.float32),
                    X_train.astype(np.float32),  # X_tilde_train (not used by admm/stg)
                    y_train,
                    X_test.astype(np.float32),
                    X_test.astype(np.float32),  # X_tilde_test
                    k,
                )

                runtime = time.time() - t0
                total_time += runtime

                auroc, auprc = evaluate_downstream(
                    X_train, y_train, X_test, y_test, scores, k, seed=s
                )
                per_seed_auroc.append(auroc)
                per_seed_auprc.append(auprc)
                print(
                    f"    Seed {i + 1}/{args.seeds}: AUROC={auroc:.4f}, time={runtime:.1f}s"
                )

            results = {
                "auroc": float(np.mean(per_seed_auroc)),
                "auroc_std": float(np.std(per_seed_auroc)),
                "auprc": float(np.mean(per_seed_auprc)),
                "auprc_std": float(np.std(per_seed_auprc)),
                "k": k,
                "time": total_time,
                "per_seed_auroc": per_seed_auroc,
                "per_seed_auprc": per_seed_auprc,
            }
            print(f"    Mean AUROC={results['auroc']:.4f}±{results['auroc_std']:.4f}")

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    Saved to {out_path}")


if __name__ == "__main__":
    main()
