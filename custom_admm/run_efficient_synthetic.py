#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run admm_input_group on synthetic datasets - efficient version.
"""

import os
import sys
import json
import numpy as np

# Add paths - use custom_admm/src for the consistent implementation
CUSTOM_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src",
)
sys.path.insert(0, CUSTOM_SRC)

from src.admm_input_group_wrapper import run_admm_input_group


def generate_ring_dataset(n_samples, n_features):
    """Generate ring dataset."""
    np.random.seed(42)
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radius = np.random.uniform(0.5, 1.0, n_samples)
    x1 = radius * np.cos(angles)
    x2 = radius * np.sin(angles)

    X = np.random.randn(n_samples, n_features)
    X[:, 0] = x1
    X[:, 1] = x2
    return X


def generate_xor_dataset(n_samples, n_features):
    """Generate XOR dataset."""
    X = np.random.rand(n_samples, n_features)
    X[:, 0] = np.random.rand(n_samples)
    X[:, 1] = np.random.rand(n_samples)
    return X


def generate_ring_xor_dataset(n_samples, n_features):
    """Generate combined ring+XOR dataset."""
    X = np.random.randn(n_samples, n_features)
    # Ring part
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radius = np.random.uniform(0.5, 1.0, n_samples)
    X[:, 0] = radius * np.cos(angles)
    X[:, 1] = radius * np.sin(angles)
    # XOR part
    X[:, 32] = np.random.rand(n_samples)
    X[:, 33] = np.random.rand(n_samples)
    return X


def generate_ring_xor_sum_dataset(n_samples, n_features):
    """Generate ring+XOR+sum dataset."""
    X = np.random.randn(n_samples, n_features)
    # Ring part
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radius = np.random.uniform(0.5, 1.0, n_samples)
    X[:, 0] = radius * np.cos(angles)
    X[:, 1] = radius * np.sin(angles)
    # XOR part
    X[:, 32] = np.random.rand(n_samples)
    X[:, 33] = np.random.rand(n_samples)
    # Sum feature
    X[:, 63] = X[:, 0] + X[:, 1] + X[:, 32] + X[:, 33]
    return X


def create_binary_labels(X, dataset_name):
    """Create binary classification labels."""
    if dataset_name == "ring":
        center = np.mean(X, axis=0)
        distances = np.linalg.norm(X - center, axis=1)
        median_dist = np.median(distances)
        y_binary = (distances > median_dist).astype(np.float32)
    elif dataset_name == "xor":
        xor_result = (X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)
        y_binary = xor_result.astype(np.float32)
    elif dataset_name == "ring+xor":
        center = np.mean(X[:, :32], axis=0)
        distances = np.linalg.norm(X[:, :32] - center, axis=1)
        ring_labels = (distances > np.median(distances)).astype(np.float32)
        xor_result = (X[:, 32] > 0.5) ^ (X[:, 33] > 0.5)
        xor_labels = xor_result.astype(np.float32)
        y_binary = (ring_labels + xor_labels > 0.5).astype(np.float32)
    else:  # ring+xor+sum
        center = np.mean(X[:, :32], axis=0)
        distances = np.linalg.norm(X[:, :32] - center, axis=1)
        ring_labels = (distances > np.median(distances)).astype(np.float32)
        xor_result = (X[:, 32] > 0.5) ^ (X[:, 33] > 0.5)
        xor_labels = xor_result.astype(np.float32)
        y_binary = (ring_labels + xor_labels > 0.5).astype(np.float32)

    return y_binary


def generate_synthetic_dataset(dataset_name, n_samples, n_features):
    """Generate synthetic dataset."""
    if dataset_name == "ring":
        return generate_ring_dataset(n_samples, n_features)
    elif dataset_name == "xor":
        return generate_xor_dataset(n_samples, n_features)
    elif dataset_name == "ring+xor":
        return generate_ring_xor_dataset(n_samples, n_features)
    else:  # ring+xor+sum
        return generate_ring_xor_sum_dataset(n_samples, n_features)


def run_synthetic_experiment(dataset_name, n_samples=500, n_features=64, seed=42):
    """Run admm_input_group on a specific synthetic dataset."""
    print(f"Running {dataset_name} (n={n_samples}, m={n_features})")

    X = generate_synthetic_dataset(dataset_name, n_samples, n_features)
    y = create_binary_labels(X, dataset_name)
    n_classes = 2

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    y_train_hat, y_hat, scores, scores2 = run_admm_input_group(
        X_train=X_train, y_train=y_train, X_test=X_test, n_classes=n_classes, seed=seed
    )

    results = {
        "method": "admm_input_group",
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "seed": seed,
        "scores": scores.tolist(),
        "scores2": scores2.tolist(),
        "y_train_hat": y_train_hat.tolist(),
        "y_hat": y_hat.tolist(),
        "n_classes": n_classes,
        "non_zero_features": int(np.sum(np.abs(scores) > 1e-6)),
    }

    return results


def main():
    """Run focused experiments."""
    # Reduced scope to avoid timeout
    datasets = ["ring", "xor"]
    n_samples = 500
    n_features_list = [64, 128]
    seeds = [42, 43]

    all_results = {}

    for dataset in datasets:
        for n_features in n_features_list:
            for seed in seeds:
                try:
                    results = run_synthetic_experiment(
                        dataset_name=dataset,
                        n_samples=n_samples,
                        n_features=n_features,
                        seed=seed,
                    )

                    key = f"{dataset}-{n_features}-seed{seed}"
                    all_results[key] = results
                    print(f"Completed: {key}")

                except Exception as e:
                    print(f"Failed {dataset}-{n_features}-seed{seed}: {e}")
                    continue

    # Save results
    output_file = "admm_input_group_consistent_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExperiments completed! Results saved to {output_file}")
    print(f"Total experiments: {len(all_results)}")


if __name__ == "__main__":
    main()
