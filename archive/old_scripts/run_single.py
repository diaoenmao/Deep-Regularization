#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run single admm_input_group experiment on ring dataset.
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


def create_binary_labels(X):
    """Create binary labels based on distance from center."""
    center = np.mean(X, axis=0)
    distances = np.linalg.norm(X - center, axis=1)
    median_dist = np.median(distances)
    y_binary = (distances > median_dist).astype(np.float32)
    return y_binary


def main():
    """Run single experiment."""
    print("Running single ring experiment...")

    # Small dataset
    X = generate_ring_dataset(200, 32)
    y = create_binary_labels(X)
    n_classes = 2

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Run admm_input_group
    y_train_hat, y_hat, scores, scores2 = run_admm_input_group(
        X_train=X_train, y_train=y_train, X_test=X_test, n_classes=n_classes, seed=42
    )

    print(f"Success! Non-zero features: {np.sum(scores > 1e-6)}")

    # Save results
    results = {
        "method": "admm_input_group",
        "dataset": "ring",
        "n_samples": 200,
        "n_features": 32,
        "seed": 42,
        "scores": scores.tolist(),
        "non_zero_features": int(np.sum(scores > 1e-6)),
    }

    with open("single_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to single_experiment_results.json")


if __name__ == "__main__":
    main()
