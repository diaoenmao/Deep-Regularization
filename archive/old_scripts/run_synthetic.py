#!/usr/bin/env python
"""
Integration script to run admm_input_group on synthetic datasets using the original
Feature-Selection-Benchmark framework.

This script:
1. Uses the original benchmark's data generation (src/data.py)
2. Calls your custom admm_input_group implementation
3. Produces results in the original benchmark format
4. Supports the same synthetic datasets as the original benchmark
"""

import sys
import os
import json
import numpy as np

# Add the original Feature-Selection-Benchmark to path for data loading
BENCH_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Feature-Selection-Benchmark"
)
sys.path.insert(0, BENCH_ROOT)

# Add custom_admm to path for your implementation
CUSTOM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUSTOM_ROOT)

from src.data import generate_dataset
from src.admm_input_group_wrapper import run_admm_input_group


def run_synthetic_experiment(
    dataset_name, n_samples=1000, n_features=64, k_true=8, seed=42
):
    """
    Run admm_input_group on a synthetic dataset.

    Args:
        dataset_name: 'ring', 'xor', 'sum', 'madelon', etc.
        n_samples: Number of samples
        n_features: Number of features
        k_true: Number of true informative features
        seed: Random seed for reproducibility

    Returns:
        dict: Results in original benchmark format
    """
    print(
        f"Running admm_input_group on {dataset_name} (n={n_samples}, m={n_features}, k={k_true})"
    )

    # Generate dataset using original benchmark's data module
    X, y, groups = generate_dataset(
        name=dataset_name,
        n_samples=n_samples,
        n_features=n_features,
        n_informative=k_true,
        seed=seed,
    )

    n_classes = len(np.unique(y))

    # Split into train/test (80/20)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Run your custom admm_input_group method
    y_train_hat, y_hat, scores, scores2 = run_admm_input_group(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        n_classes=n_classes,
        seed=seed,
    )

    # Format results to match original benchmark structure
    results = {
        "method": "admm_input_group",
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "k_true": k_true,
        "seed": seed,
        "scores": scores.tolist(),
        "scores2": scores2.tolist(),
        "y_train_hat": y_train_hat.tolist(),
        "y_hat": y_hat.tolist(),
        "n_classes": n_classes,
    }

    return results


def main():
    """Run synthetic experiments and save results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run admm_input_group on synthetic datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ring",
        help="Dataset name (ring, xor, sum, madelon)",
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_features", type=int, default=64)
    parser.add_argument("--k_true", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="results.json", help="Output file path"
    )

    args = parser.parse_args()

    results = run_synthetic_experiment(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        n_features=args.n_features,
        k_true=args.k_true,
        seed=args.seed,
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
