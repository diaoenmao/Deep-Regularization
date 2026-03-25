#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: Rapid Experimental Validation for ADMM Improvements

快速验证以下改进：
1. 早停机制 (Early Stopping)
2. 高斯噪声 (Gaussian Noise)
3. 优化器对比 (Adagrad vs Adam)

使用方法:
    python run_phase1_experiments.py --gpu 0 --quick

输出:
    - results/phase1_experiments_YYYYMMDD_HHMMSS.json
    - results/phase1_summary_YYYYMMDD_HHMMSS.txt
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch
from datetime import datetime

# Add paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data import generate_dataset
from admm_input_group_wrapper import GatedFeatureSelectionMLP, _train_input_group, _extract_feature_importance
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Configuration
SEED = 0
N_SAMPLES = 1000
N_FOLDS = 6

# 快速实验配置 - 只测试部分维度
QUICK_CONFIG = {
    "xor": (2, [8, 128, 1024]),  # 代表低/中/高维度
    "ring": (2, [32, 512]),
    "ring+xor": (4, [16, 256]),
    "ring+xor+sum": (6, [64]),
}

# 完整配置
FULL_CONFIG = {
    "xor": (2, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    "ring": (2, [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    "ring+xor": (4, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    "ring+xor+sum": (6, [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
}

# 实验配置
EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "原始 ADMM (无早停，无噪声)",
        "config": {
            "gaussian_noise": 0.0,
            "use_early_stopping": False,
            "optimizer": "adam",
            "epochs": 500,
        }
    },
    {
        "name": "early_stop",
        "desc": "ADMM + 早停 (patience=66)",
        "config": {
            "gaussian_noise": 0.0,
            "use_early_stopping": True,
            "patience": 66,
            "optimizer": "adam",
            "epochs": 500,
        }
    },
    {
        "name": "noise",
        "desc": "ADMM + 高斯噪声 (0.747)",
        "config": {
            "gaussian_noise": 0.7466805127272365,
            "use_early_stopping": False,
            "optimizer": "adam",
            "epochs": 500,
        }
    },
    {
        "name": "early_stop_noise",
        "desc": "ADMM + 早停 + 高斯噪声",
        "config": {
            "gaussian_noise": 0.7466805127272365,
            "use_early_stopping": True,
            "patience": 66,
            "optimizer": "adam",
            "epochs": 500,
        }
    },
    {
        "name": "long_patience",
        "desc": "ADMM + 早停 (patience=100) + 噪声",
        "config": {
            "gaussian_noise": 0.7466805127272365,
            "use_early_stopping": True,
            "patience": 100,
            "optimizer": "adam",
            "epochs": 500,
        }
    },
    {
        "name": "full_improvement",
        "desc": "ADMM + 早停 (patience=66) + 噪声 (0.747) + val_split=0.3",
        "config": {
            "gaussian_noise": 0.7466805127272365,
            "use_early_stopping": True,
            "patience": 66,
            "optimizer": "adam",
            "epochs": 500,
            "val_split": 0.3,
        }
    },
]


def evaluate_fold(model, X_train, y_train, X_test, correct, k, device, train_config):
    """Evaluate a single fold with given model configuration."""
    _train_input_group(
        model, X_train, y_train, n_classes=2,
        lr=train_config.get("lr", 0.005),
        C=0.05,
        epochs=train_config.get("epochs", 500),
        warmup_epochs=120,
        batch_size=64,
        rho_init=200.0 if X_train.shape[1] >= 512 else 50.0,
        device=device,
        # Phase 1 experimental parameters
        optimizer_type=train_config.get("optimizer", "adam"),
        use_early_stopping=train_config.get("use_early_stopping", False),
        patience=train_config.get("patience", 66),
        val_split=train_config.get("val_split", 0.2),
    )

    scores = _extract_feature_importance(model, X_train)

    top_ranked = np.argsort(np.abs(scores))
    best_k = sum(i in correct for i in top_ranked[-k:]) / k

    return best_k, scores.tolist()


def run_single_experiment(ds_name, k, n_features, exp_config, device, seed=SEED):
    """Run single experiment for one dataset and dimension."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, X_tilde, y = generate_dataset(ds_name, N_SAMPLES, n_features)
    X = 2.0 * X - 1.0
    X_tilde = 2.0 * X_tilde - 1.0

    splits = list(KFold(n_splits=N_FOLDS).split(X))
    best_ks = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
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

        # Create model with experiment config
        model = GatedFeatureSelectionMLP(
            input_size=n_features,
            n_classes=2,
            latent_size=58,
            n_hidden_layers=5,
            gaussian_noise=exp_config.get("gaussian_noise", 0.0),
            dropout=0.04308691548552568,
            feat_drop=0.6,
            activation="mish",
        )

        best_k, _ = evaluate_fold(
            model, X_train_s, y_train, X_test, correct, k, device, exp_config
        )
        best_ks.append(best_k)

    return {
        "mean": np.mean(best_ks),
        "std": np.std(best_ks),
        "per_fold": best_ks,
    }


def run_all_experiments(use_quick=True, device="cpu", output_dir="results"):
    """Run all experiments and save results."""
    config = QUICK_CONFIG if use_quick else FULL_CONFIG

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"phase1_experiments_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"phase1_summary_{timestamp}.txt")

    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "quick_mode": use_quick,
            "device": device,
            "n_samples": N_SAMPLES,
            "n_folds": N_FOLDS,
        },
        "experiments": {},
    }

    total_runs = sum(len(dims) for dims in config.values()) * len(EXPERIMENTS)
    run_count = 0
    start_time = time.time()

    print("=" * 80)
    print("Phase 1: Rapid Experimental Validation")
    print("=" * 80)
    print(f"Mode: {'QUICK' if use_quick else 'FULL'} | Device: {device}")
    print(f"Total runs: {total_runs}")
    print("=" * 80)

    for exp in EXPERIMENTS:
        print(f"\n>>> Experiment: {exp['name']} - {exp['desc']}")
        print("-" * 60)

        all_results["experiments"][exp["name"]] = {
            "desc": exp["desc"],
            "config": exp["config"],
            "datasets": {},
        }

        for ds_name, (k, dimensions) in config.items():
            print(f"\n  Dataset: {ds_name} (k={k})")
            ds_results = {}

            for n_features in dimensions:
                run_count += 1
                elapsed = time.time() - start_time
                eta = elapsed / run_count * (total_runs - run_count) if run_count > 0 else 0

                print(f"    m={n_features} [{run_count}/{total_runs}, ETA: {eta/60:.1f}min]...", end=" ", flush=True)

                result = run_single_experiment(ds_name, k, n_features, exp["config"], device)
                ds_results[str(n_features)] = result

                print(f"best-k={result['mean']:.1%} (+/- {result['std']:.1%})")

            all_results["experiments"][exp["name"]]["datasets"][ds_name] = {
                "k": k,
                "dimensions": ds_results,
            }

            # Save intermediate results
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # Generate summary
    generate_summary(all_results, summary_file)

    print("\n" + "=" * 80)
    print("Phase 1 Complete!")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)

    return all_results


def generate_summary(results, output_file):
    """Generate human-readable summary."""
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 1: Experimental Summary\n")
        f.write("=" * 80 + "\n\n")

        for exp_name, exp_data in results["experiments"].items():
            f.write(f"\n## {exp_name}: {exp_data['desc']}\n\n")
            f.write("| Dataset | m | Best-k (mean ± std) |\n")
            f.write("|---------|-----|-------------------|\n")

            for ds_name, ds_data in exp_data["datasets"].items():
                for m, res in ds_data["dimensions"].items():
                    f.write(f"| {ds_name} | {m} | {res['mean']*100:.1f}% ± {res['std']*100:.1f}% |\n")

        # Comparison table
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Comparison: Average Best-k by Dataset\n")
        f.write("=" * 80 + "\n\n")

        f.write("| Experiment | XOR | RING | RING+XOR | RING+XOR+SUM | OVERALL |\n")
        f.write("|------------|-----|------|----------|--------------|---------|\n")

        for exp_name, exp_data in results["experiments"].items():
            row = f"| {exp_name}"
            overall_avg = []

            for ds_name in ["xor", "ring", "ring+xor", "ring+xor+sum"]:
                ds_data = exp_data["datasets"].get(ds_name, {})
                dims = ds_data.get("dimensions", {})
                if dims:
                    avg = np.mean([r["mean"] for r in dims.values()])
                    overall_avg.append(avg)
                    row += f" | {avg*100:.1f}%"
                else:
                    row += " | -"

            if overall_avg:
                row += f" | {np.mean(overall_avg)*100:.1f}% |"
            else:
                row += " | - |"

            f.write(row + "\n")

    print(f"Summary written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Rapid Experimental Validation")
    parser.add_argument("--quick", action="store_true", help="Quick mode (test fewer dimensions)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    # Determine device
    if args.gpu is not None and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Run experiments
    run_all_experiments(
        use_quick=args.quick,
        device=device,
        output_dir=args.output_dir
    )
