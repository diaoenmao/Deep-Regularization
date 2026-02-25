# -*- coding: utf-8 -*-
"""Run a single dataset+method combination. Used for incremental runs."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_modern_admm_stg import load_dataset, evaluate_downstream, SEED, N_SEEDS
from src.core import run_fs_method

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'external-data')

def run_one(ds, method):
    seeds = [SEED + i for i in range(N_SEEDS)]
    out_path = os.path.join(RESULTS_PATH, f'{ds}-{method}.json')
    if os.path.exists(out_path):
        print(f'{ds}-{method}: already exists, skipping')
        return

    X_train, y_train, X_test, y_test, n_classes = load_dataset(ds)
    n_features = X_train.shape[1]
    k = int(round(0.5 * n_features))
    print(f'{ds}-{method}: n_train={X_train.shape[0]}, n_features={n_features}, k={k}')

    per_seed_auroc, per_seed_auprc = [], []
    total_time = 0
    for i, s in enumerate(seeds):
        t0 = time.time()
        np.random.seed(s)
        torch.manual_seed(s)
        _, _, scores, _ = run_fs_method(
            ds, method,
            X_train.astype(np.float32), X_train.astype(np.float32),
            y_train, X_test.astype(np.float32), X_test.astype(np.float32), k)
        runtime = time.time() - t0
        total_time += runtime
        auroc, auprc = evaluate_downstream(X_train, y_train, X_test, y_test, scores, k, seed=s)
        per_seed_auroc.append(auroc)
        per_seed_auprc.append(auprc)
        print(f'  Seed {i+1}/{N_SEEDS}: AUROC={auroc:.4f}, time={runtime:.1f}s')

    results = {
        'auroc': float(np.mean(per_seed_auroc)),
        'auroc_std': float(np.std(per_seed_auroc)),
        'auprc': float(np.mean(per_seed_auprc)),
        'auprc_std': float(np.std(per_seed_auprc)),
        'k': k, 'time': total_time,
        'per_seed_auroc': per_seed_auroc,
        'per_seed_auprc': per_seed_auprc,
    }
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: {out_path}  Mean AUROC={results["auroc"]:.4f}+-{results["auroc_std"]:.4f}')

if __name__ == '__main__':
    ds = sys.argv[1]
    method = sys.argv[2]
    run_one(ds, method)
