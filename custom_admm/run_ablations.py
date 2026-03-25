"""
Ablation study runner for SADMM-FS.
  - 4A: Ratio Norm vs L1 (lasso_input_group) on all synthetic tasks
  - 4B: Dropout rate sweep on XOR m=128 and Ring m=64
  - 4C: Ring investigation with dropout=0.0 vs 0.7

Usage:
  python run_ablations.py --all
  python run_ablations.py --l1-ablation
  python run_ablations.py --dropout-ablation
  python run_ablations.py --ring-investigation
"""
import sys, os, json, argparse, time
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data import generate_dataset
from src.core import run_fs_method
from src.admm_input_group_wrapper import (
    run_admm_input_group, GatedFeatureSelectionMLP, _train_input_group, _HPARAMS,
)

SEED = 0
N_SAMPLES = 1000
RESULTS_PATH = os.path.join(ROOT, "results", "ablations")
os.makedirs(RESULTS_PATH, exist_ok=True)


def eval_synthetic(method, ds_name, k_true, ns, hp_overrides=None):
    """Run 6-fold CV on a synthetic dataset, return per-dimension results."""
    results = []
    for n_features in ns:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, n_features)
        X = 2.0 * X - 1.0
        X_tilde = 2.0 * X_tilde - 1.0
        k = k_true

        splits = list(KFold(n_splits=6).split(X))
        best_ks = []

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            X_tilde_train = X_tilde[train_idx]
            X_tilde_test = X_tilde[test_idx]
            y_train = y[train_idx]

            idx = np.arange(n_features)
            np.random.shuffle(idx)
            X_train, X_test = X_train[:, idx], X_test[:, idx]
            correct = set(np.where(idx < k)[0].tolist())

            if hp_overrides:
                _, _, scores, _ = run_admm_input_group(
                    X_train, y_train, X_test, n_classes=2,
                    hp_overrides=hp_overrides
                )
            else:
                _, _, scores, _ = run_fs_method(
                    ds_name, method, X_train, X_tilde_train,
                    y_train, X_test, X_tilde_test, k
                )
            if scores is not None:
                ranked = np.argsort(np.abs(scores))
                best_ks.append(sum(i in correct for i in ranked[-k:]) / k)

        bk = np.mean(best_ks) if best_ks else 0
        bk_std = np.std(best_ks) if best_ks else 0
        results.append({"m": n_features, "best_k": bk, "best_k_std": bk_std})
        print(f"    m={n_features:5d}  best-k={bk:.1%}±{bk_std:.1%}")
    return results


def run_l1_ablation():
    """Phase 4A: Compare Ratio Norm vs L1 on all synthetic tasks."""
    print("\n" + "="*60)
    print("  ABLATION 4A: Ratio Norm vs L1")
    print("="*60)

    datasets = [
        ("xor",          2, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        ("ring",         2, [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        ("ring+xor",     4, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        ("ring+xor+sum", 6, [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ]

    all_results = {}
    for ds_name, k_true, ns in datasets:
        print(f"\n  {ds_name} (L1 / lasso_input_group):")
        res = eval_synthetic("lasso_input_group", ds_name, k_true, ns)
        avg = np.mean([r["best_k"] for r in res])
        all_results[ds_name] = {"details": res, "avg_best_k": avg}
        print(f"  Average best-k: {avg:.1%}")

    with open(os.path.join(RESULTS_PATH, "l1_ablation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {os.path.join(RESULTS_PATH, 'l1_ablation.json')}")
    return all_results


def run_dropout_ablation():
    """Phase 4B: Dropout rate sweep on XOR m=128 and Ring m=64."""
    print("\n" + "="*60)
    print("  ABLATION 4B: Dropout Rate Sweep")
    print("="*60)

    dropout_rates = [0.0, 0.3, 0.5, 0.7, 0.9]
    tasks = [
        ("xor", 2, [128]),
        ("ring", 2, [64]),
    ]

    all_results = {}
    for ds_name, k_true, ns in tasks:
        task_results = {}
        for dr in dropout_rates:
            print(f"\n  {ds_name} m={ns[0]}, dropout={dr}:")
            res = eval_synthetic(
                "admm_input_group", ds_name, k_true, ns,
                hp_overrides={"feat_drop": dr}
            )
            task_results[str(dr)] = res[0]
        all_results[ds_name] = task_results

    with open(os.path.join(RESULTS_PATH, "dropout_ablation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {os.path.join(RESULTS_PATH, 'dropout_ablation.json')}")
    return all_results


def run_ring_investigation():
    """Phase 4C: Ring at m=64 with dropout=0.0 vs 0.7."""
    print("\n" + "="*60)
    print("  ABLATION 4C: Ring Investigation")
    print("="*60)

    results = {}
    for dr in [0.0, 0.7]:
        print(f"\n  Ring m=64, dropout={dr}:")
        res = eval_synthetic(
            "admm_input_group", "ring", 2, [64],
            hp_overrides={"feat_drop": dr}
        )
        results[f"dropout_{dr}"] = res[0]

    with open(os.path.join(RESULTS_PATH, "ring_investigation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.join(RESULTS_PATH, 'ring_investigation.json')}")
    return results


def run_convergence_save():
    """Run one XOR m=128 experiment and save the ADMM convergence log to JSON."""
    print("\n" + "="*60)
    print("  Saving ADMM convergence log (XOR m=128)")
    print("="*60)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    X, X_tilde, y = generate_dataset("xor", N_SAMPLES, 128)
    X = 2.0 * X - 1.0
    X_train, y_train = X[:800], y[:800]

    # Standardise
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    hp = _HPARAMS["admm_input_group"]
    model = GatedFeatureSelectionMLP(
        input_size=128, n_classes=2, latent_size=32,
        n_hidden_layers=2, feat_drop=hp.get("feat_drop", 0.7), activation="mish",
    )
    _train_input_group(
        model, X_train_s, y_train, n_classes=2,
        lr=hp["lr"], C=hp["C"], epochs=hp["epochs"],
        warmup_epochs=hp.get("warmup_epochs", 120),
        rho_init=hp.get("rho_init", 200.0),
        use_ratio_norm=True, use_admm=True,
    )

    if hasattr(model, "convergence_log") and model.convergence_log:
        log = model.convergence_log
        data = {
            "epochs":  [e for e, _, _ in log],
            "primal":  [p for _, p, _ in log],
            "dual":    [d for _, _, d in log],
        }
        out_path = os.path.join(RESULTS_PATH, "convergence_log.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {out_path} ({len(log)} entries)")
    else:
        print("ERROR: No convergence log found on model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--l1-ablation", action="store_true")
    parser.add_argument("--dropout-ablation", action="store_true")
    parser.add_argument("--ring-investigation", action="store_true")
    parser.add_argument("--save-convergence", action="store_true")
    args = parser.parse_args()

    if args.all or args.save_convergence:
        run_convergence_save()
    if args.all or args.l1_ablation:
        run_l1_ablation()
    if args.all or args.dropout_ablation:
        run_dropout_ablation()
    if args.all or args.ring_investigation:
        run_ring_investigation()

    if not any([args.all, args.l1_ablation, args.dropout_ablation,
                args.ring_investigation, args.save_convergence]):
        print("No ablation selected. Use --all, --l1-ablation, --dropout-ablation, --ring-investigation, or --save-convergence.")
