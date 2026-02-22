"""
Run ADMM-Gate (Ratio Norm) through the official 6-fold CV benchmark.
Covers: xor, ring, ring+xor, ring+xor+sum, dag  (synthetic)
        + MADELON (real, NIPS 2003)
"""
import sys, os, time, json, argparse
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data import generate_dataset
from src.dag import load_dag_dataset, generate_dag_dataset
from src.knockoff import generate_gaussian_knockoffs
from src.core import run_fs_method

parser = argparse.ArgumentParser()
parser.add_argument("--dag-seeds", type=int, default=1, help="Number of random DAG seeds")
parser.add_argument("--madelon-seeds", type=int, default=5, help="Number of RF seeds for MADELON")
args = parser.parse_args()

SEED = 0
N_SAMPLES = 1000
METHOD = "admm_input_group"
DATA_PATH = os.path.join(ROOT, "data")
RESULTS_PATH = os.path.join(ROOT, "results")
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# ── Synthetic datasets ─────────────────────────────────────────────────
datasets_config = [
    ("xor",          2, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring",         2, [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor",     4, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor+sum", 6, [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
]

summary = {}

for ds_name, k_true, ns in datasets_config:
    outfile = os.path.join(RESULTS_PATH, f"{METHOD}-{ds_name}-{N_SAMPLES}.txt")
    print(f"\n{'='*60}")
    print(f"  {ds_name}  k={k_true}")
    print(f"{'='*60}")

    ds_results = []
    with open(outfile, "w") as f:
        f.write(f"Dataset\tADMM_InputGroup_bestK\tADMM_InputGroup_bestK_std"
                f"\tADMM_InputGroup_best2K\tADMM_InputGroup_best2K_std"
                f"\tADMM_InputGroup_TrainAUC\tADMM_InputGroup_TrainAUPRC"
                f"\tADMM_InputGroup_AUC\tADMM_InputGroup_AUPRC\n")

        for n_features in ns:
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, n_features)
            X = 2.0 * X - 1.0
            X_tilde = 2.0 * X_tilde - 1.0
            k = k_true

            splits = list(KFold(n_splits=6).split(X))
            best_ks, best_2ks = [], []

            t0 = time.time()
            for train_idx, test_idx in splits:
                X_train, X_test = X[train_idx], X[test_idx]
                X_tilde_train = X_tilde[train_idx]
                X_tilde_test = X_tilde[test_idx]
                y_train = y[train_idx]

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
            bk_std = np.std(best_ks) if best_ks else 0
            b2k_std = np.std(best_2ks) if best_2ks else 0
            print(f"  m={n_features:5d}  best-k={bk:.1%}±{bk_std:.1%}  best-2k={b2k:.1%}±{b2k_std:.1%}  ({elapsed:.0f}s)")
            row_name = f"{ds_name}_{n_features}_{N_SAMPLES}"
            f.write(f"{row_name}\t{bk}\t{bk_std}\t{b2k}\t{b2k_std}\t\t\t\t\n")
            ds_results.append({"m": n_features, "best_k": bk, "best_k_std": bk_std,
                               "best_2k": b2k, "best_2k_std": b2k_std,
                               "per_fold_bestk": best_ks, "per_fold_best2k": best_2ks})

    avg_bk = np.mean([r["best_k"] for r in ds_results])
    summary[ds_name] = {"avg_best_k": avg_bk, "details": ds_results}
    print(f"  Average best-k: {avg_bk:.1%}")
    print(f"  Saved: {outfile}")


# ── DAG dataset ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  dag  (2000 features, {args.dag_seeds} seed(s))")
print(f"{'='*60}")

dag_outfile = os.path.join(RESULTS_PATH, f"{METHOD}-dag-{N_SAMPLES}.txt")

all_dag_bk, all_dag_b2k, all_dag_bk2, all_dag_b2k2 = [], [], [], []

for dag_seed in range(args.dag_seeds):
    np.random.seed(dag_seed)
    torch.manual_seed(dag_seed)

    # Generate a fresh DAG for each seed (bypass cache)
    if args.dag_seeds == 1:
        X, X_tilde, y, k, k2 = load_dag_dataset(DATA_PATH)
    else:
        X, y, k, k2 = generate_dag_dataset(1000, 2000, 20, 1000, density=0.004, sigma=0.2)
        X_tilde = generate_gaussian_knockoffs(X)

    X = StandardScaler().fit_transform(X)
    X_tilde = StandardScaler().fit_transform(X_tilde)
    n_features = X.shape[1]

    splits = list(KFold(n_splits=6).split(X))
    best_ks, best_2ks = [], []
    best_k2s, best_2k2s = [], []

    t0 = time.time()
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        X_tilde_train = X_tilde[train_idx]
        X_tilde_test = X_tilde[test_idx]
        y_train = y[train_idx]

        idx = np.arange(n_features)
        np.random.shuffle(idx)
        X_train, X_test = X_train[:, idx], X_test[:, idx]
        correct_k = set(np.where(idx < k)[0].tolist())
        correct_k2 = set(np.where(idx < k2)[0].tolist())

        _, _, scores, scores2 = run_fs_method(
            "dag", METHOD, X_train, X_tilde_train,
            y_train, X_test, X_tilde_test, k
        )
        if scores is not None:
            ranked = np.argsort(np.abs(scores))
            best_ks.append(sum(i in correct_k for i in ranked[-k:]) / k)
            best_k2s.append(sum(i in correct_k2 for i in ranked[-k2:]) / k2)
            ranked2 = np.argsort(np.abs(scores2))
            best_2ks.append(sum(i in correct_k for i in ranked2[-2*k:]) / k)
            best_2k2s.append(sum(i in correct_k2 for i in ranked2[-2*k2:]) / k2)

    elapsed = time.time() - t0
    bk = np.mean(best_ks) if best_ks else 0
    b2k = np.mean(best_2ks) if best_2ks else 0
    bk2 = np.mean(best_k2s) if best_k2s else 0
    b2k2 = np.mean(best_2k2s) if best_2k2s else 0
    all_dag_bk.append(bk)
    all_dag_b2k.append(b2k)
    all_dag_bk2.append(bk2)
    all_dag_b2k2.append(b2k2)
    print(f"  seed={dag_seed}  k={k}, k2={k2}  best-k={bk:.1%}  best-2k={b2k:.1%}  best-k2={bk2:.1%}  best-2k2={b2k2:.1%}  ({elapsed:.0f}s)")

# Aggregate across seeds
dag_bk_mean, dag_bk_std = np.mean(all_dag_bk), np.std(all_dag_bk)
dag_b2k_mean, dag_b2k_std = np.mean(all_dag_b2k), np.std(all_dag_b2k)
dag_bk2_mean, dag_bk2_std = np.mean(all_dag_bk2), np.std(all_dag_bk2)
dag_b2k2_mean, dag_b2k2_std = np.mean(all_dag_b2k2), np.std(all_dag_b2k2)

with open(dag_outfile, "w") as f:
    f.write(f"Dataset\tADMM_InputGroup_bestK\tADMM_InputGroup_bestK_std"
            f"\tADMM_InputGroup_best2K\tADMM_InputGroup_best2K_std"
            f"\tADMM_InputGroup_bestK2\tADMM_InputGroup_bestK2_std"
            f"\tADMM_InputGroup_best2K2\tADMM_InputGroup_best2K2_std\n")
    f.write(f"dag_2000_{N_SAMPLES}\t{dag_bk_mean}\t{dag_bk_std}\t{dag_b2k_mean}\t{dag_b2k_std}"
            f"\t{dag_bk2_mean}\t{dag_bk2_std}\t{dag_b2k2_mean}\t{dag_b2k2_std}\n")

print(f"  AGGREGATE ({args.dag_seeds} seeds):")
print(f"    best-k={dag_bk_mean:.1%}±{dag_bk_std:.1%}  best-2k={dag_b2k_mean:.1%}±{dag_b2k_std:.1%}")
print(f"    best-k2={dag_bk2_mean:.1%}±{dag_bk2_std:.1%}  best-2k2={dag_b2k2_mean:.1%}±{dag_b2k2_std:.1%}")
summary["dag"] = {
    "best_k": dag_bk_mean, "best_k_std": dag_bk_std,
    "best_2k": dag_b2k_mean, "best_2k_std": dag_b2k_std,
    "best_k2": dag_bk2_mean, "best_k2_std": dag_bk2_std,
    "best_2k2": dag_b2k2_mean, "best_2k2_std": dag_b2k2_std,
    "per_seed": [{"bk": a, "b2k": b, "bk2": c, "b2k2": d}
                 for a, b, c, d in zip(all_dag_bk, all_dag_b2k, all_dag_bk2, all_dag_b2k2)]
}
print(f"  Saved: {dag_outfile}")


# ── MADELON (real dataset, NIPS 2003) ──────────────────────────────────
print(f"\n{'='*60}")
print(f"  MADELON  (500 features, 96% decoy)")
print(f"{'='*60}")

madelon_dir = os.path.join(DATA_PATH, "madelon", "MADELON")
madelon_avail = os.path.exists(os.path.join(madelon_dir, "madelon_train.data"))

if not madelon_avail:
    print("  MADELON data not found. Downloading...")
    os.makedirs(madelon_dir, exist_ok=True)
    import urllib.request
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON"
    files = [
        ("madelon_train.data", f"{base_url}/madelon_train.data"),
        ("madelon_valid.data", f"{base_url}/madelon_valid.data"),
        ("madelon_train.labels", f"{base_url}/madelon_train.labels"),
    ]
    # valid labels are at a different path
    valid_labels_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels"
    madelon_label_dir = os.path.join(DATA_PATH, "madelon")
    try:
        for fname, url in files:
            dest = os.path.join(madelon_dir, fname)
            if not os.path.exists(dest):
                print(f"    Downloading {fname}...")
                urllib.request.urlretrieve(url, dest)
        vl_dest = os.path.join(madelon_label_dir, "madelon_valid.labels")
        if not os.path.exists(vl_dest):
            print(f"    Downloading madelon_valid.labels...")
            urllib.request.urlretrieve(valid_labels_url, vl_dest)
        madelon_avail = True
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Skipping MADELON.")

if madelon_avail:
    def load_dense(fp):
        data = []
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append([int(x) for x in line.split()])
        return np.array(data, dtype=float)

    def load_labels(fp):
        data = []
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(int(line))
        y = np.array(data, dtype=int)
        return (y > 0).astype(int)

    X_train_m = load_dense(os.path.join(madelon_dir, "madelon_train.data"))
    X_test_m = load_dense(os.path.join(madelon_dir, "madelon_valid.data"))
    y_train_m = load_labels(os.path.join(madelon_dir, "madelon_train.labels"))
    y_test_m = load_labels(os.path.join(DATA_PATH, "madelon", "madelon_valid.labels"))

    scaler = StandardScaler()
    X_train_m = scaler.fit_transform(X_train_m)
    X_test_m = scaler.transform(X_test_m)

    n_features_m = X_train_m.shape[1]  # 500
    n_classes_m = 2
    k_mad = int(round(0.04 * n_features_m))  # 4% informative = 20
    X_tilde_m = X_train_m  # dummy knockoffs

    t0 = time.time()
    _, _, scores_m, _ = run_fs_method(
        "madelon", METHOD, X_train_m, X_tilde_m,
        y_train_m, X_test_m, X_test_m, k_mad, _2k=False
    )
    elapsed = time.time() - t0

    # Evaluate: train RF on top-k features with multiple seeds
    top_idx = np.argsort(np.abs(scores_m))[-k_mad:]
    aurocs, auprcs = [], []
    for rf_seed in range(args.madelon_seeds):
        rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=rf_seed)
        rf.fit(X_train_m[:, top_idx], y_train_m)
        y_hat_m = rf.predict_proba(X_test_m[:, top_idx])[:, 1]
        aurocs.append(roc_auc_score(y_test_m, y_hat_m))
        auprcs.append(average_precision_score(y_test_m, y_hat_m))

    auroc_mean, auroc_std = np.mean(aurocs), np.std(aurocs)
    auprc_mean, auprc_std = np.mean(auprcs), np.std(auprcs)

    print(f"  k={k_mad}, top features: {sorted(top_idx[:10].tolist())}...")
    print(f"  AUROC={auroc_mean:.4f}±{auroc_std:.4f}  AUPRC={auprc_mean:.4f}±{auprc_std:.4f}  ({elapsed:.0f}s, {args.madelon_seeds} RF seeds)")

    madelon_results = {
        "auroc": auroc_mean, "auroc_std": auroc_std,
        "auprc": auprc_mean, "auprc_std": auprc_std,
        "k": k_mad, "time": elapsed,
        "per_seed_auroc": aurocs, "per_seed_auprc": auprcs
    }
    summary["madelon"] = madelon_results

    madelon_out = os.path.join(RESULTS_PATH, "external-data")
    os.makedirs(madelon_out, exist_ok=True)
    with open(os.path.join(madelon_out, f"madelon-{METHOD}.json"), "w") as f:
        json.dump(madelon_results, f, indent=2)
    print(f"  Saved: {os.path.join(madelon_out, f'madelon-{METHOD}.json')}")


# ── Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
for ds, data in summary.items():
    if ds == "madelon":
        print(f"  {ds:18s}  AUROC={data['auroc']:.4f}±{data.get('auroc_std',0):.4f}  AUPRC={data['auprc']:.4f}±{data.get('auprc_std',0):.4f}")
    elif ds == "dag":
        print(f"  {ds:18s}  best-k={data['best_k']:.1%}±{data.get('best_k_std',0):.1%}  best-k2={data['best_k2']:.1%}±{data.get('best_k2_std',0):.1%}")
    else:
        print(f"  {ds:18s}  avg best-k={data['avg_best_k']:.1%}")

# Save full summary
with open(os.path.join(RESULTS_PATH, f"{METHOD}_full_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nFull summary saved to results/{METHOD}_full_summary.json")
print("\nDone.")
