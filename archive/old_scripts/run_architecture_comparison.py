#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证假设：使用 LassoNet 风格的小模型架构能否提升 ADMM 效果

LassoNet 配置：
- hidden_dims = (32, 32)  # 2 层，每层 32 单元
- n_iters = (30, 30)
- dropout = 0

对比配置：
- Baseline: 5 层×58 单元 (当前 ADMM)
- Small: 2 层×32 单元 (LassoNet 风格)
"""
import sys
import os
import json
import time
import numpy as np
import torch
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data import generate_dataset
from admm_input_group_wrapper import GatedFeatureSelectionMLP, _train_input_group, _extract_feature_importance
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SEED = 0
N_SAMPLES = 1000
N_FOLDS = 6

# 快速测试配置 - 选择代表性维度
QUICK_CONFIG = {
    "xor": (2, [8, 128, 1024]),
    "ring": (2, [32, 512]),
    "ring+xor": (4, [16, 256]),
    "ring+xor+sum": (6, [64]),
}

# 架构配置
ARCHITECTURES = [
    {
        "name": "baseline_5layer",
        "desc": "Baseline: 5 层×58 单元 (当前 ADMM)",
        "config": {
            "n_hidden_layers": 5,
            "latent_size": 58,
            "gaussian_noise": 0.0,
            "dropout": 0.043,
        }
    },
    {
        "name": "small_2layer",
        "desc": "Small: 2 层×32 单元 (LassoNet 风格)",
        "config": {
            "n_hidden_layers": 2,
            "latent_size": 32,
            "gaussian_noise": 0.0,
            "dropout": 0.0,  # LassoNet 无 dropout
        }
    },
    {
        "name": "small_2layer_noise",
        "desc": "Small+ 噪声：2 层×32 单元 + 高斯噪声",
        "config": {
            "n_hidden_layers": 2,
            "latent_size": 32,
            "gaussian_noise": 0.747,
            "dropout": 0.0,
        }
    },
    {
        "name": "medium_3layer",
        "desc": "Medium: 3 层×48 单元 (折中方案)",
        "config": {
            "n_hidden_layers": 3,
            "latent_size": 48,
            "gaussian_noise": 0.0,
            "dropout": 0.043,
        }
    },
]


def run_fold(ds_name, k, n_features, arch_config, fold_idx, device):
    """Run single fold with given architecture."""
    np.random.seed(SEED + fold_idx)
    torch.manual_seed(SEED + fold_idx)

    X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, n_features)
    X = 2.0 * X - 1.0

    splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X))
    train_idx, test_idx = splits[fold_idx]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]

    # Shuffle features
    idx = np.arange(n_features)
    np.random.shuffle(idx)
    X_train, X_test = X_train[:, idx], X_test[:, idx]
    correct = set(np.where(idx < k)[0].tolist())

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Create model
    model = GatedFeatureSelectionMLP(
        input_size=n_features,
        n_classes=2,
        latent_size=arch_config["latent_size"],
        n_hidden_layers=arch_config["n_hidden_layers"],
        gaussian_noise=arch_config.get("gaussian_noise", 0.0),
        dropout=arch_config.get("dropout", 0.043),
        feat_drop=0.6,
        activation="mish",
    )

    # Train
    _train_input_group(
        model, X_train_s, y_train, n_classes=2,
        lr=0.005, C=0.05, epochs=500, warmup_epochs=120,
        batch_size=64, rho_init=200.0 if n_features >= 512 else 50.0,
        device=device,
        use_early_stopping=False,
    )

    # Evaluate
    scores = _extract_feature_importance(model, X_train_s)
    top_ranked = np.argsort(np.abs(scores))
    best_k = sum(i in correct for i in top_ranked[-k:]) / k

    return best_k


def run_architecture_comparison(use_quick=True, device="cuda"):
    """Run architecture comparison."""
    config = QUICK_CONFIG if use_quick else FULL_CONFIG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = os.path.join(ROOT, "results", f"architecture_comparison_{timestamp}.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "quick_mode": use_quick,
            "device": device,
            "n_samples": N_SAMPLES,
            "n_folds": N_FOLDS,
        },
        "architectures": {},
    }

    total_runs = sum(len(dims) for dims in config.values()) * len(ARCHITECTURES) * N_FOLDS
    run_count = 0
    start_time = time.time()

    print("=" * 80)
    print("Architecture Comparison: LassoNet-style Small Model vs Baseline")
    print("=" * 80)
    print(f"Device: {device} | Mode: {'QUICK' if use_quick else 'FULL'}")
    print(f"Total runs: {total_runs} ({len(ARCHITECTURES)} architectures × {len(config)} datasets × {N_FOLDS} folds)")
    print("=" * 80)

    for arch in ARCHITECTURES:
        print(f"\n>>> Architecture: {arch['name']} - {arch['desc']}")
        print("-" * 60)

        all_results["architectures"][arch["name"]] = {
            "desc": arch["desc"],
            "config": arch["config"],
            "datasets": {},
        }

        for ds_name, (k, dimensions) in config.items():
            ds_results = {}

            for n_features in dimensions:
                fold_results = []

                for fold_idx in range(N_FOLDS):
                    run_count += 1
                    elapsed = time.time() - start_time
                    eta = elapsed / run_count * (total_runs - run_count) if run_count > 0 else 0

                    if fold_idx == 0:
                        print(f"  {ds_name} m={n_features} fold={fold_idx+1}...", end="", flush=True)
                    else:
                        print(f"{fold_idx+1}", end="", flush=True)

                    best_k = run_fold(ds_name, k, n_features, arch["config"], fold_idx, device)
                    fold_results.append(best_k)

                mean_bk = np.mean(fold_results)
                std_bk = np.std(fold_results)
                ds_results[str(n_features)] = {
                    "mean": mean_bk,
                    "std": std_bk,
                    "per_fold": fold_results,
                }

                print(f" → {mean_bk*100:.1f}% (+/- {std_bk*100:.1f}%)  [ETA: {eta/60:.0f}min]")

            all_results["architectures"][arch["name"]]["datasets"][ds_name] = {
                "k": k,
                "dimensions": ds_results,
            }

            # Save intermediate
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # Generate summary
    generate_summary(all_results, results_file.replace(".json", "_summary.txt"))

    print("\n" + "=" * 80)
    print(f"Complete! Results: {results_file}")
    print("=" * 80)

    return all_results


def generate_summary(results, output_file):
    """Generate summary report."""
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Architecture Comparison Summary\n")
        f.write("=" * 80 + "\n\n")

        # Comparison table
        f.write("Average Best-k by Architecture and Dataset:\n\n")
        f.write("| Architecture | XOR | RING | RING+XOR | RING+XOR+SUM | OVERALL |\n")
        f.write("|--------------|-----|------|----------|--------------|---------|\n")

        for arch_name, arch_data in results["architectures"].items():
            row = f"| {arch_name[:12]:<12} "
            overall_avg = []

            for ds_name in ["xor", "ring", "ring+xor", "ring+xor+sum"]:
                ds_data = arch_data["datasets"].get(ds_name, {})
                dims = ds_data.get("dimensions", {})
                if dims:
                    avg = np.mean([r["mean"] for r in dims.values()])
                    overall_avg.append(avg)
                    row += f"| {avg*100:>5.1f}% "
                else:
                    row += "|   -   "

            if overall_avg:
                row += f"| {np.mean(overall_avg)*100:>6.1f}% |"
            else:
                row += "|   -   |"

            f.write(row + "\n")

        # Detailed breakdown
        f.write("\n\nDetailed Breakdown:\n\n")
        for arch_name, arch_data in results["architectures"].items():
            f.write(f"\n## {arch_name}: {arch_data['desc']}\n\n")
            for ds_name, ds_data in arch_data["datasets"].items():
                f.write(f"### {ds_name}\n")
                for m, res in ds_data["dimensions"].items():
                    f.write(f"  m={m}: {res['mean']*100:.1f}% (+/- {res['std']*100:.1f}%)\n")

    print(f"Summary written to: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID")
    args = parser.parse_args()

    if args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    run_architecture_comparison(use_quick=args.quick, device=device)
