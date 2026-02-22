"""Run admm_input_group through the official benchmark protocol and save results."""
import sys, os, time
import numpy as np
import torch
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data import generate_dataset
from src.core import run_fs_method

SEED = 0
N_SAMPLES = 1000
METHOD = "admm_input_group"

datasets_config = [
    ("xor",          2, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring",         2, [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor",     4, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor+sum", 6, [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
]

results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

for ds_name, k, ns in datasets_config:
    outfile = os.path.join(results_dir, f"{METHOD}-{ds_name}-{N_SAMPLES}.txt")
    print(f"\n{'='*60}")
    print(f"  {ds_name}  k={k}")
    print(f"{'='*60}")

    with open(outfile, "w") as f:
        f.write(f"Dataset\tADMM_InputGroup_bestK\tADMM_InputGroup_best2K\tADMM_InputGroup_TrainAUC\tADMM_InputGroup_TrainAUPRC\tADMM_InputGroup_AUC\tADMM_InputGroup_AUPRC\n")

        for n_features in ns:
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, n_features)
            X = 2.0 * X - 1.0
            X_tilde = 2.0 * X_tilde - 1.0

            splits = list(KFold(n_splits=6).split(X))
            best_ks = []
            best_2ks = []

            t0 = time.time()
            for train_idx, test_idx in splits:
                X_train, X_test = X[train_idx], X[test_idx]
                X_tilde_train, X_tilde_test = X_tilde[train_idx], X_tilde[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                idx = np.arange(n_features)
                np.random.shuffle(idx)
                X_train, X_test = X_train[:, idx], X_test[:, idx]
                correct = set(np.where(idx < k)[0].tolist())

                _, _, scores, scores2 = run_fs_method(
                    ds_name, METHOD, X_train, X_tilde_train,
                    y_train, X_test, X_tilde_test, k
                )

                if scores is not None:
                    ranked = np.argsort(np.abs(scores))
                    best_ks.append(sum(i in correct for i in ranked[-k:]) / k)
                    ranked2 = np.argsort(np.abs(scores2))
                    best_2ks.append(sum(i in correct for i in ranked2[-2*k:]) / k)

            elapsed = time.time() - t0
            bk = np.mean(best_ks) if best_ks else 0
            b2k = np.mean(best_2ks) if best_2ks else 0
            print(f"  m={n_features:5d}  best-k={bk:.1%}  best-2k={b2k:.1%}  ({elapsed:.0f}s)")
            row_name = f"{ds_name}_{n_features}_{N_SAMPLES}"
            f.write(f"{row_name}\t{bk}\t{b2k}\t\t\t\t\n")

    print(f"  Saved: {outfile}")

print("\nDone.")
