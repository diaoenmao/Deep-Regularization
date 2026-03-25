# -*- coding: utf-8 -*-
"""
Run SADMM-FS (admm_input_group) and STG on NIPS 2003 real-world datasets
with multi-seed evaluation (5 seeds).

Datasets:
  - MADELON  (500 features, binary classification)
  - Arcene   (10,000 features, mass spectrometry)
  - Gisette  (5,000 features, handwriting)
  - Dexter   (20,000 features, text categorization)

Usage:
    python run_realworld_admm.py [--method admm_input_group|stg] [--seeds 5]
"""

import argparse
import json
import os
import time
import tracemalloc

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.admm_input_group_wrapper import run_admm_input_group
from src.stg_wrapper import run_stg_fs

SEED = 0xCAFE
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(ROOT, 'results', 'external-data')
os.makedirs(RESULTS_PATH, exist_ok=True)

# Dataset configs: name -> (decoy_fraction, n_features_for_sparse)
DATASETS = {
    'madelon': 0.96,
    'arcene':  0.3,
    'gisette': 0.3,
    'dexter':  0.5,
}


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
                    assert 0 <= j < m
                    xs[j] = value
                data.append(xs)
    return np.asarray(data, dtype=int)


def load_dataset(name):
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
        raise ValueError(f'Unknown dataset: {name}')
    y_train = load_nips2003_labels(os.path.join(sub_folder, f'{name}_train.labels'))
    y_test = load_nips2003_labels(os.path.join(folder, f'{name}_valid.labels'))
    return X_train, y_train, X_test, y_test


def evaluate_downstream(X_train, y_train, X_test, y_test, scores, k):
    """Select top-k features, train RF, return AUROC and AUPRC."""
    idx = np.argsort(scores)[-k:]
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
    model.fit(X_train[:, idx], y_train)
    y_hat = model.predict_proba(X_test[:, idx])
    y_hat_score = y_hat[:, 1] if y_hat.shape[1] == 2 else y_hat
    auroc = roc_auc_score(y_test, y_hat_score)
    auprc = average_precision_score(y_test, y_hat_score)
    return auroc, auprc


def run_single_seed(method, dataset_name, X_train, y_train, X_test, y_test,
                    n_classes, k, hp_overrides, seed_val):
    """Run one seed for a given method and return (auroc, auprc, time)."""
    t0 = time.time()

    if method == 'admm_input_group':
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)
        _, _, scores, _ = run_admm_input_group(
            X_train.astype(np.float32),
            y_train, X_test.astype(np.float32),
            n_classes,
            hp_overrides=hp_overrides,
            seed=seed_val,
        )
    elif method == 'stg':
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)
        _, _, scores, _ = run_stg_fs(
            X_train.astype(np.float32),
            y_train, X_test.astype(np.float32),
            n_classes,
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    runtime = time.time() - t0
    auroc, auprc = evaluate_downstream(X_train, y_train, X_test, y_test, scores, k)
    return auroc, auprc, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='admm_input_group',
                        choices=['admm_input_group', 'stg'])
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    seeds = [42 + i * 1000 for i in range(args.seeds)]

    for dataset_name, decoy_fraction in DATASETS.items():
        out_path = os.path.join(RESULTS_PATH, f'{dataset_name}-{args.method}.json')

        print(f'\n{"="*60}')
        print(f'Dataset: {dataset_name} | Method: {args.method} | Seeds: {args.seeds}')
        print(f'{"="*60}')

        try:
            X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        except FileNotFoundError as e:
            print(f'  SKIPPED — data not found: {e}')
            continue

        n_features = X_train.shape[1]
        n_classes = len(set(y_train))
        k = int(round((1.0 - decoy_fraction) * n_features))

        print(f'  n_train={len(X_train)}, n_test={len(X_test)}, '
              f'n_features={n_features}, k={k}, n_classes={n_classes}')

        hp_overrides = {}
        if n_features >= 10000 and args.method == 'admm_input_group':
            hp_overrides = {'epochs': 600, 'warmup_epochs': 150}

        per_seed_auroc = []
        per_seed_auprc = []
        total_time = 0

        for i, s in enumerate(seeds):
            auroc, auprc, runtime = run_single_seed(
                args.method, dataset_name,
                X_train, y_train, X_test, y_test,
                n_classes, k, hp_overrides, s)
            per_seed_auroc.append(auroc)
            per_seed_auprc.append(auprc)
            total_time += runtime
            print(f'  Seed {i+1}/{args.seeds} (s={s}): AUROC={auroc:.4f}, AUPRC={auprc:.4f}, time={runtime:.1f}s')

        results = {
            'auroc': float(np.mean(per_seed_auroc)),
            'auroc_std': float(np.std(per_seed_auroc)),
            'auprc': float(np.mean(per_seed_auprc)),
            'auprc_std': float(np.std(per_seed_auprc)),
            'k': k,
            'time': total_time,
            'per_seed_auroc': per_seed_auroc,
            'per_seed_auprc': per_seed_auprc,
        }

        print(f'  Mean AUROC={results["auroc"]:.4f}±{results["auroc_std"]:.4f}')

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Saved to {out_path}')


if __name__ == '__main__':
    main()
