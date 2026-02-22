# -*- coding: utf-8 -*-
"""
P1.6: Downstream task performance experiments.

Part A: Synthetic downstream AUROC
  - For each FS method × synthetic task × dimension:
    select top-k features, train RF on selected, report test AUROC.

Part B: Real-world AUROC-vs-k curves
  - For each FS method × NIPS dataset:
    vary k, train RF on top-k features, report test AUROC.

Usage:
    python run_downstream.py --part A
    python run_downstream.py --part B
    python run_downstream.py --part both
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.core import run_fs_method
from src.data import generate_dataset
from run_realworld_admm import load_dataset, DATASETS as REAL_DATASETS

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(ROOT, 'results', 'downstream')
os.makedirs(RESULTS_PATH, exist_ok=True)

SEED = 0xCAFE

# Methods to evaluate
METHODS = ['admm_input_group', 'stg', 'rf', 'treeshap', 'lassonet', 'mi', 'relief']

# Synthetic configs
SYNTH_TASKS = {
    'xor':           {'k': 2, 'dims': [8, 32, 128, 512, 2048]},
    'ring':          {'k': 2, 'dims': [8, 32, 128, 512, 2048]},
    'ring+xor':      {'k': 4, 'dims': [8, 32, 128, 512, 2048]},
    'ring+xor+sum':  {'k': 6, 'dims': [8, 32, 128, 512, 2048]},
}

N_SAMPLES = 1000
N_FOLDS = 6


def downstream_auroc(X_train, y_train, X_test, y_test, top_idx):
    """Train RF on selected features, return test AUROC."""
    if len(top_idx) == 0:
        return 0.5
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
    rf.fit(X_train[:, top_idx], y_train)
    y_prob = rf.predict_proba(X_test[:, top_idx])
    if y_prob.shape[1] == 2:
        y_score = y_prob[:, 1]
    else:
        y_score = y_prob
    return roc_auc_score(y_test, y_score)


def run_part_a():
    """Synthetic downstream AUROC: top-k features -> RF -> test AUROC."""
    results = {}

    for task_name, cfg in SYNTH_TASKS.items():
        k = cfg['k']
        results[task_name] = {}

        for m in cfg['dims']:
            print(f'\n--- {task_name} m={m} ---')
            results[task_name][m] = {}

            # Generate data
            X, X_tilde, y = generate_dataset(task_name, N_SAMPLES, m)
            X = 2.0 * X - 1.0
            X_tilde = 2.0 * X_tilde - 1.0

            splits = list(KFold(n_splits=N_FOLDS, shuffle=False).split(X))

            for method in METHODS:
                fold_aurocs = []
                for fold_i, (train_idx, test_idx) in enumerate(splits):
                    X_train, X_test = X[train_idx], X[test_idx]
                    X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Permute features (same as main benchmark)
                    idx = np.arange(m)
                    np.random.seed(SEED + fold_i)
                    np.random.shuffle(idx)
                    X_train_p, X_test_p = X_train[:, idx], X_test[:, idx]
                    X_tilde_train_p = X_tilde_train[:, idx]
                    X_tilde_test_p = X_tilde_test[:, idx]

                    try:
                        _, _, scores, _ = run_fs_method(
                            task_name, method,
                            X_train_p, X_tilde_train_p, y_train,
                            X_test_p, X_tilde_test_p, k
                        )
                    except Exception as e:
                        print(f'  {method} fold {fold_i} FAILED: {e}')
                        scores = None

                    if scores is not None:
                        top_k_idx = np.argsort(np.abs(scores))[-k:]
                        auroc = downstream_auroc(
                            X_train_p, y_train, X_test_p, y_test, top_k_idx)
                        fold_aurocs.append(auroc)

                if fold_aurocs:
                    mean_auroc = float(np.mean(fold_aurocs))
                    std_auroc = float(np.std(fold_aurocs))
                    results[task_name][m][method] = {
                        'auroc': mean_auroc, 'auroc_std': std_auroc
                    }
                    print(f'  {method}: {mean_auroc:.4f}±{std_auroc:.4f}')
                else:
                    results[task_name][m][method] = {'auroc': None, 'auroc_std': None}
                    print(f'  {method}: FAILED')

    out_path = os.path.join(RESULTS_PATH, 'synthetic_downstream.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')


def run_part_b():
    """Real-world AUROC-vs-k: vary k, train RF on top-k, report AUROC."""
    # Methods that produce feature scores (not just classifiers)
    rw_methods = ['admm_input_group', 'stg', 'rf', 'treeshap', 'mi', 'relief']

    # k fractions to evaluate
    k_fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    results = {}

    for dataset_name, decoy_frac in REAL_DATASETS.items():
        print(f'\n{"="*60}')
        print(f'Dataset: {dataset_name}')
        print(f'{"="*60}')

        try:
            X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        except FileNotFoundError as e:
            print(f'  SKIPPED: {e}')
            continue

        n_features = X_train.shape[1]
        n_classes = len(set(y_train))
        results[dataset_name] = {}

        k_values = sorted(set(
            max(1, int(round(f * n_features))) for f in k_fractions
        ))
        print(f'  k values: {k_values}')

        for method in rw_methods:
            print(f'\n  Method: {method}')
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            # Get feature scores
            try:
                if method == 'admm_input_group':
                    from src.admm_lasso_wrapper import run_admm_lasso_fs
                    hp_overrides = {}
                    if n_features >= 10000:
                        hp_overrides = {'epochs': 600, 'warmup_epochs': 150}
                    _, _, scores, _ = run_admm_lasso_fs(
                        'admm_input_group',
                        X_train.astype(np.float32), y_train,
                        X_test.astype(np.float32), n_classes,
                        hp_overrides=hp_overrides, seed=SEED,
                    )
                elif method == 'stg':
                    from src.stg_wrapper import run_stg_fs
                    _, _, scores, _ = run_stg_fs(
                        X_train.astype(np.float32), y_train,
                        X_test.astype(np.float32), n_classes,
                    )
                elif method == 'rf':
                    clf = RandomForestClassifier(
                        n_estimators=500, n_jobs=-1, random_state=SEED)
                    clf.fit(X_train, y_train)
                    scores = clf.feature_importances_
                elif method == 'treeshap':
                    clf = RandomForestClassifier(
                        n_estimators=500, n_jobs=-1, random_state=SEED)
                    clf.fit(X_train, y_train)
                    import shap
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_train)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    sv = np.array(shap_values)
                    if sv.ndim == 3:
                        sv = sv[:, :, 1]  # take class 1
                    scores = np.abs(sv).mean(axis=0)
                elif method == 'mi':
                    from sklearn.feature_selection import mutual_info_classif
                    scores = mutual_info_classif(X_train, y_train)
                elif method == 'relief':
                    from src.core import relief
                    scores = relief(X_train, y_train)
                else:
                    continue
            except Exception as e:
                print(f'    FAILED to get scores: {e}')
                continue

            # Evaluate at each k
            method_results = []
            for k in k_values:
                top_idx = np.argsort(np.abs(scores))[-k:]
                auroc = downstream_auroc(X_train, y_train, X_test, y_test, top_idx)
                method_results.append({'k': k, 'auroc': float(auroc)})
                print(f'    k={k}: AUROC={auroc:.4f}')

            results[dataset_name][method] = method_results

    out_path = os.path.join(RESULTS_PATH, 'realworld_auroc_vs_k.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, default='both',
                        choices=['A', 'B', 'both'])
    args = parser.parse_args()

    if args.part in ('A', 'both'):
        run_part_a()
    if args.part in ('B', 'both'):
        run_part_b()
