# -*- coding: utf-8 -*-
"""
Test script for TODO implementations.

Tests:
1. Iterative Run variants
2. Polynomial Expansion + Selection
3. Transformer Pretrain
"""

import numpy as np
import torch

# Test imports
print("Testing imports...")

try:
    from src.iterative_run import (
        iterative_hard_pruning,
        iterative_gate_refinement,
        train_with_gradual_admm,
    )
    print("[OK] iterative_run imported")
except Exception as e:
    print(f"[FAIL] iterative_run failed: {e}")

try:
    from src.polynomial_expansion import (
        compute_polynomial_expansion,
        compute_fourier_features,
        PolynomialFeatureSelectionModel,
        run_polynomial_fs_experiment,
    )
    print("[OK] polynomial_expansion imported")
except Exception as e:
    print(f"[FAIL] polynomial_expansion failed: {e}")

try:
    from src.transformer_pretrain import (
        MaskedFeaturePretrainer,
        pretrain_transformer,
        run_transformer_pretrain_experiment,
        get_recommended_config,
    )
    print("[OK] transformer_pretrain imported")
except Exception as e:
    print(f"[FAIL] transformer_pretrain failed: {e}")


def test_polynomial_expansion():
    """Test polynomial feature expansion."""
    print("\n" + "=" * 50)
    print("Testing Polynomial Expansion")
    print("=" * 50)

    # Create simple test data
    np.random.seed(42)
    X = np.random.randn(100, 5)

    # Test degree 2
    X_exp, names, poly = compute_polynomial_expansion(X, degree=2)
    print(f"Original: {X.shape[1]} features")
    print(f"Expanded (degree=2): {X_exp.shape[1]} features")
    print(f"Feature names: {names[:5]}...")

    # Test model creation
    model = PolynomialFeatureSelectionModel(
        input_size=5,
        n_classes=2,
        degree=2,
        selection_mode="group",
    )
    print(f"Model created with {model.expanded_size} expanded features")

    # Test forward pass
    X_t = torch.tensor(X[:10], dtype=torch.float32)
    with torch.no_grad():
        out = model(X_t)
    print(f"Forward pass: input {X_t.shape} -> output {out.shape}")

    # Test Fourier features
    X_fourier = compute_fourier_features(X_t, n_bands=5)
    print(f"Fourier features: {X_t.shape} -> {X_fourier.shape}")

    print("[OK] Polynomial expansion tests passed")


def test_transformer_pretrain():
    """Test transformer pretraining."""
    print("\n" + "=" * 50)
    print("Testing Transformer Pretrain")
    print("=" * 50)

    # Create simple test data
    np.random.seed(42)
    X = np.random.randn(100, 10)

    # Create model
    model = MaskedFeaturePretrainer(
        input_size=10,
        d_model=16,
        n_heads=4,
        n_layers=1,
        mask_ratio=0.3,
    )

    # Test forward pass
    X_t = torch.tensor(X[:8], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        recon, loss, mask = model(X_t)

    print(f"Input: {X_t.shape}")
    print(f"Reconstructed: {recon.shape}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Mask ratio: {mask.float().mean().item():.2%}")

    # Test encoding
    encoded = model.encode(X_t)
    print(f"Encoded (CLS): {encoded.shape}")

    # Test quick pretrain
    print("\nQuick pretrain test (5 epochs)...")
    pretrained = pretrain_transformer(
        X,
        d_model=16,
        n_heads=4,
        n_layers=1,
        lr=1e-4,
        epochs=5,
        verbose=True,
    )

    print("✓ Transformer pretrain tests passed")


def test_iterative_run():
    """Test iterative run variants."""
    print("\n" + "=" * 50)
    print("Testing Iterative Run")
    print("=" * 50)

    # Create test data (XOR-like)
    np.random.seed(42)
    n_samples = 200
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    # XOR on first 2 features
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    print(f"Data: {X.shape}, classes: {np.bincount(y)}")

    # Test with simple MLP
    class SimpleMLP(torch.nn.Module):
        def __init__(self, input_size, n_classes=2, hidden=32):
            super().__init__()
            self.gate = torch.nn.Parameter(torch.zeros(input_size))
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden),
                torch.nn.Mish(),
                torch.nn.Linear(hidden, n_classes),
            )
            self.first_linear = self.net[0]

        def forward(self, x):
            g = self.gate
            return self.net(x * g)

        def get_feature_scores(self):
            return torch.abs(self.gate)

    def simple_train(model, X, y, n_classes, epochs=50, device="cpu", **kwargs):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = crit(model(X_t), y_t)
            loss.backward()
            opt.step()

    def get_scores(model):
        return model.get_feature_scores()

    print("\nTesting iterative_gate_refinement...")
    alive_features, history = iterative_gate_refinement(
        model_class=SimpleMLP,
        model_kwargs={"hidden": 32},
        X_train=X,
        y_train=y,
        n_classes=2,
        train_fn=simple_train,
        threshold=0.01,
        n_iterations=3,
        verbose=True,
    )
    print(f"Final features: {alive_features[:5]}... ({len(alive_features)} total)")

    print("✓ Iterative run tests passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TODO Implementations Test Suite")
    print("=" * 60)

    test_polynomial_expansion()
    test_transformer_pretrain()
    test_iterative_run()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()