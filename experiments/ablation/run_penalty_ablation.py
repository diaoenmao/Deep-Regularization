# -*- coding: utf-8 -*-
"""
run_penalty_ablation.py — Compare adaptive (λ_j = C/s_j) vs uniform (λ_j = C) penalty.

Runs on DAG + NIPS 2003 datasets with 5 seeds each.

Usage:
    python run_penalty_ablation.py
    python run_penalty_ablation.py --datasets dag madelon
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.admm_input_group_wrapper import (
    GatedFeatureSelectionMLP, _train_input_group, _extract_feature_importance,
)
from src.dag import load_dag_dataset

RESULTS_DIR = os.path.join(ROOT, "results", "penalty_ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 0xCAFE
N_SEEDS = 5
HP = dict(lr=1e-3, C=0.05, epochs=500, warmup_epochs=120, rho_init=200)

DATASETS = ['dag', 'madelon', 'arcene', 'gisette', 'dexter']


def load_nips2003_labels(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                data.append(int(line))
    y = np.asarray(data, dtype=int)
    return (y > 0).astype(int)


def load_nips2003_dense_matrix(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            elements = line.rstrip().split(' ')
            if len(elements) > 0:
                data.append([int(x) for x in elements])
    return np.asarray(data, dtype=int)


def load_nips2003_sparse_matrix(filepath, m):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            elements = line.rstrip().split(' ')
            if len(elements) > 0:
                xs = np.zeros(m, dtype=int)
                for el in elements:
                    j, value = el.split(':')
                    j, value = int(j) - 1, int(value)
                    xs[j] = value
                data.append(xs)
    return np.asarray(data, dtype=int)


def load_nips_dataset(name):
    DATA_PATH = os.path.join(ROOT, 'data')
    folder = os.path.join(DATA_PATH, name)
    sub_folder = os.path.join(DATA_PATH, name, name.upper())
    if name in {'arcene', 'gisette', 'madelon'}:
        X_train = load_nips2003_dense_matrix(os.path.join(sub_folder, f'{name}_train.data'))
        X_test = load_nips2003_dense_matrix(os.path.join(sub_folder, f'{name}_valid.data'))
    elif name == 'dexter':
        X_train = load_nips2003_sparse_matrix(os.path.join(sub_folder, f'{name}_train.data'), 20000)
        X_test = load_nips2003_sparse_matrix(os.path.join(sub_folder, f'{name}_valid.data'), 20000)
    else:
        raise ValueError(f'Unknown: {name}')
    y_train = load_nips2003_labels(os.path.join(sub_folder, f'{name}_train.labels'))
    y_test = load_nips2003_labels(os.path.join(folder, f'{name}_valid.labels'))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(float))
    X_test = scaler.transform(X_test.astype(float))
    return X_train, y_train, X_test, y_test


def evaluate_dag_metrics(scores, k, k2):
    ranking = np.argsort(scores)[::-1]
    true_k = set(range(k))
    true_k2 = set(range(k2))
    topk = set(ranking[:k].tolist())
    topk2 = set(ranking[:k2].tolist())
    top2k2 = set(ranking[:2 * k2].tolist())
    return {
        "bestK": len(topk & true_k) / k,
        "bestK2": len(topk2 & true_k2) / k2,
        "best2K2": len(top2k2 & true_k2) / k2,
    }


def evaluate_downstream(X_train, y_train, X_test, y_test, scores, k, seed):
    idx = np.argsort(scores)[-k:]
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    model.fit(X_train[:, idx], y_train)
    y_hat = model.predict_proba(X_test[:, idx])[:, 1]
    auroc = roc_auc_score(y_test, y_hat)
    auprc = average_precision_score(y_test, y_hat)
    return auroc, auprc


def run_one(X, y, n_classes, seed, uniform_penalty):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X_s, y_s = X[idx], y[idx]

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GatedFeatureSelectionMLP(
        input_size=X.shape[1], n_classes=n_classes,
        latent_size=32, n_hidden_layers=2, feat_drop=0.7, activation="mish",
    )
    _train_input_group(model, X_s, y_s, n_classes, **HP, uniform_penalty=uniform_penalty)
    scores = _extract_feature_importance(model, X_s)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: penalty_ablation.json)')
    args = parser.parse_args()

    seeds = [SEED + i for i in range(N_SEEDS)]
    all_results = {}

    for ds in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        if ds == 'dag':
            X, X_tilde, y, k, k2 = load_dag_dataset(os.path.join(ROOT, 'data'))
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            n_classes = 2
        else:
            X_train, y_train, X_test, y_test = load_nips_dataset(ds)
            X = X_train
            y = y_train
            n_classes = 2
            decoy = {'madelon': 0.96, 'arcene': 0.3, 'gisette': 0.3, 'dexter': 0.5}
            k = int(round((1 - decoy[ds]) * X.shape[1]))

        for mode_name, uniform in [('adaptive', False), ('uniform', True)]:
            print(f"\n  Penalty: {mode_name}")
            per_seed = []

            for i, s in enumerate(seeds):
                t0 = time.time()
                scores = run_one(X, y, n_classes, s, uniform_penalty=uniform)
                elapsed = time.time() - t0

                if ds == 'dag':
                    metrics = evaluate_dag_metrics(scores, k, k2)
                    print(f"    Seed {i+1}: bestK={metrics['bestK']*100:.1f}% "
                          f"bestK2={metrics['bestK2']*100:.1f}% ({elapsed:.1f}s)")
                else:
                    auroc, auprc = evaluate_downstream(
                        X_train, y_train, X_test, y_test, scores, k, s)
                    metrics = {'auroc': auroc, 'auprc': auprc}
                    print(f"    Seed {i+1}: AUROC={auroc:.4f} ({elapsed:.1f}s)")

                metrics['time'] = elapsed
                per_seed.append(metrics)

            all_results[f"{ds}_{mode_name}"] = per_seed

            # Print summary for this mode
            if ds == 'dag':
                bk = [r['bestK'] for r in per_seed]
                bk2 = [r['bestK2'] for r in per_seed]
                print(f"    Mean: bestK={np.mean(bk)*100:.1f}±{np.std(bk)*100:.1f}% "
                      f"bestK2={np.mean(bk2)*100:.1f}±{np.std(bk2)*100:.1f}%")
            else:
                aurocs = [r['auroc'] for r in per_seed]
                print(f"    Mean: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}")

    out_name = args.output or "penalty_ablation.json"
    out_path = os.path.join(RESULTS_DIR, out_name)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
