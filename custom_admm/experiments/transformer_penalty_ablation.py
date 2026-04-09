"""
Quick experiment: Transformer backbone with adaptive penalty vs uniform penalty.

Goal: Verify whether the transformer backbone conclusion (it fails) holds
when using the same penalty policy as gated_mlp.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.admm_input_group_wrapper import _Scaler, _train_input_group
from src.data import generate_dataset
from src.mentor_models import GatedTokenTransformerFS


def run_single_fold(
    model,
    X_train,
    y_train,
    X_test,
    *,
    uniform_penalty: bool,
    device: str,
    epochs: int = 240,
    warmup_epochs: int = 60,
):
    """Train and return predictions + gate scores."""
    scaler = _Scaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = model.to(device)
    _train_input_group(
        model,
        X_train_s,
        y_train,
        n_classes=2,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        uniform_penalty=uniform_penalty,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        x_t = torch.FloatTensor(X_test_s).to(device)
        logits = model(x_t).cpu().numpy().flatten()
        y_hat = 1.0 / (1.0 + np.exp(-logits))

    gate = model.gate.detach().cpu().numpy()
    if hasattr(model, "gate_from_parameter"):
        gate = model.gate_from_parameter(torch.from_numpy(gate)).numpy()

    return y_hat, np.abs(gate)


def evaluate_best_k(gate_scores, k_true, m):
    """Return best-k recovery rate."""
    top_k = set(np.argsort(gate_scores)[-k_true:])
    true_features = set(range(k_true))
    return len(top_k & true_features) / k_true


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Focus on key tasks from the report
    # (dataset_type, k_true, m)
    tasks = [
        ("xor", 2, 128),
        ("ring", 2, 128),
        ("ring+xor", 4, 256),
    ]

    results = {}

    for dataset_type, k_true, m in tasks:
        task_name = f"{dataset_type}_m{m}"
        print(f"\n{'='*60}")
        print(f"Task: {task_name} (k={k_true})")
        print(f"{'='*60}")

        # Generate data once
        X, X_tilde, y = generate_dataset(dataset_type, n_samples=1000, n_features=m)

        kf = KFold(n_splits=6, shuffle=True, random_state=42)

        for penalty_type, uniform in [("uniform", True), ("adaptive", False)]:
            print(f"\n--- Penalty: {penalty_type} ---")
            best_k_scores = []
            auc_scores = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = GatedTokenTransformerFS(
                    input_size=m,
                    n_classes=2,
                    d_model=32,
                    n_heads=4,
                    n_layers=2,
                    ff_dim=128,
                    feat_drop=0.6,
                    bounded_gate=False,
                    dropout=0.1,
                )

                y_hat, gate = run_single_fold(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    uniform_penalty=uniform,
                    device=device,
                )

                best_k = evaluate_best_k(gate, k_true, m)
                auc = roc_auc_score(y_test, y_hat)

                best_k_scores.append(best_k)
                auc_scores.append(auc)
                print(f"  Fold {fold}: best_k={best_k:.4f}, AUC={auc:.4f}")

            mean_best_k = np.mean(best_k_scores)
            mean_auc = np.mean(auc_scores)
            print(f"  Mean: best_k={mean_best_k:.4f}, AUC={mean_auc:.4f}")

            key = f"{task_name}_{penalty_type}"
            results[key] = {
                "task": task_name,
                "penalty": penalty_type,
                "mean_best_k": mean_best_k,
                "mean_auc": mean_auc,
                "per_fold_best_k": best_k_scores,
                "per_fold_auc": auc_scores,
            }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':<20} {'Uniform best-k':>15} {'Adaptive best-k':>15} {'Delta':>10}")
    print("-" * 60)
    for dataset_type, k_true, m in tasks:
        task_name = f"{dataset_type}_m{m}"
        uniform_k = results[f"{task_name}_uniform"]["mean_best_k"]
        adaptive_k = results[f"{task_name}_adaptive"]["mean_best_k"]
        delta = adaptive_k - uniform_k
        print(f"{task_name:<20} {uniform_k:>15.4f} {adaptive_k:>15.4f} {delta:>+10.4f}")

    # Save results
    out_path = os.path.join(ROOT, "results", "mentor_axes", "transformer_penalty_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("If adaptive penalty improves transformer significantly, the previous")
    print("comparison was unfair. If it doesn't (or makes it worse), the original")
    print("conclusion stands: transformer backbone fails on this task regardless.")


if __name__ == "__main__":
    main()