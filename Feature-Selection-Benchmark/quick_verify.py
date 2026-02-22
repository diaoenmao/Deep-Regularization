"""Quick 5-seed verification: xor, ring, ring+xor+sum at m=128."""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import generate_dataset
from src.core import run_fs_method

SEEDS = [0, 1, 2, 3, 4]
N = 1000
M = 128
METHOD = "admm_input_group"

datasets = [("xor", 2), ("ring", 2), ("ring+xor+sum", 6)]

for ds_name, k in datasets:
    results = []
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        X, X_tilde, y = generate_dataset(ds_name, N, M)
        X = 2.0 * X - 1.0
        X_tilde = 2.0 * X_tilde - 1.0

        t0 = time.time()
        _, _, scores, _ = run_fs_method(
            ds_name, METHOD, X, X_tilde, y, X, X_tilde, k
        )
        elapsed = time.time() - t0

        ranked = np.argsort(np.abs(scores))
        top_k = set(ranked[-k:].tolist())
        correct = set(range(k))
        recall = len(top_k & correct) / k
        results.append(recall)
        print(f"  {ds_name}  seed={seed}  recall={recall:.0%}  ({elapsed:.0f}s)")

    avg = np.mean(results)
    print(f"  >>> {ds_name} avg: {avg:.1%}  ({results})\n")
