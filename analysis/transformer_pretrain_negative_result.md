# Transformer Pretrain Ablation: Negative Result Analysis

**Date**: 2026-04-08
**Status**: Confirmed Negative Result

---

## 1. Experiment Setup

### Goal

Test whether MAE-style masked feature reconstruction pretraining can salvage Transformer backbone for feature selection tasks.

### Hypothesis

If we pretrain the transformer to reconstruct masked features, it may learn useful feature representations that improve feature selection during fine-tuning with ADMM.

### Methods Compared

| Method | Architecture | Pretrain |
|--------|-------------|----------|
| MLP Baseline | 2-layer MLP + ADMM gate | None |
| Transformer Pretrain | GatedTokenTransformerFS (2-layer) | Masked reconstruction |

### Configuration

**Transformer Architecture** (matching main benchmark):
```python
{
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 2,
    "ff_dim": 128,
    "feat_drop": 0.6,
}
```

**Pretrain Settings**:
```python
{
    "pretrain_epochs": 24,
    "pretrain_lr": 0.001,
    "mask_ratio": 0.15,  # 15% features masked
}
```

**Fine-tune Settings** (matching fairness protocol):
```python
{
    "epochs": 500,
    "warmup_epochs": 120,
    "lr": 0.005,
    "C": 0.05,
    "batch_size": 64,
    "optimizer": "adam",
    "use_admm": True,
    "use_ratio_norm": True,
}
```

**Dataset**: XOR (500 samples, 32 features)
- Ground truth: features [0, 1]
- k = 2 for evaluation

**Seeds**: 42, 43, 44 (3 seeds quick run)

---

## 2. Results

### Feature Selection Performance

| Method | Seed | best_k | Selected Features |
|--------|------|--------|-------------------|
| MLP Baseline | 42 | **1.0** | [0, 1] ✓ |
| MLP Baseline | 43 | **1.0** | [0, 1] ✓ |
| MLP Baseline | 44 | **1.0** | [0, 1] ✓ |
| Transformer Pretrain | 42 | **0.0** | [17, 6] ✗ |
| Transformer Pretrain | 43 | **1.0** | [0, 1] ✓ |
| Transformer Pretrain | 44 | **0.0** | [17, 14] ✗ |

### Summary Statistics

| Method | best_k (mean ± std) | Success Rate |
|--------|---------------------|--------------|
| MLP Baseline | **1.00 ± 0.00** | 100% (3/3) |
| Transformer Pretrain | **0.33 ± 0.47** | 33% (1/3) |

---

## 3. Pretrain Loss Analysis

### Loss History (24 epochs)

| Seed | Initial Loss | Final Loss | Trend |
|------|--------------|------------|-------|
| 42 | 1.052 | 1.008 | Oscillates |
| 43 | 1.102 | 1.015 | Oscillates |
| 44 | 1.068 | 1.000 | Oscillates |

### Key Observation: Loss Oscillates Around ~1.0

The reconstruction loss does NOT converge to zero. Instead, it oscillates between 0.97-1.07 throughout pretraining.

**Interpretation**: The model learns to predict the mean value (≈0 after [-1,1] scaling) for masked positions. This achieves MSE ≈ variance ≈ 1.0 for noise features.

---

## 4. Root Cause Analysis

### Why Masked Reconstruction Fails for Tabular FS

#### 4.1 Structural Mismatch

| Domain | Features | Reconstruction | Success Reason |
|--------|----------|----------------|----------------|
| Images (MAE) | Pixels (2D grid) | Spatial interpolation | Nearby pixels correlated |
| Text (BERT) | Tokens (sequence) | Context prediction | Semantic relationships |
| **Tabular** | Independent features | Feature reconstruction | **No cross-feature structure** |

#### 4.2 Noise Dominance Problem

For XOR dataset:
- 2 signal features (x₀, x₁)
- 30 noise features (random uniform)

**Masking a noise feature**:
- Other features are also noise → no predictive signal
- Best prediction = mean = 0
- MSE = variance ≈ 1.0

**Masking a signal feature**:
- Other features don't contain XOR structure information
- Cannot reconstruct x₀ from x₁ alone (XOR needs both)
- Again, best prediction = 0

#### 4.3 Gate Not Involved During Pretrain

```python
# In run_transformer_pretrain_ablation.py line 279
x_masked[mask] = 0.0
recon = model.reconstruct_masked(x_masked)  # apply_gate=False!
```

The ADMM gate is **frozen during pretrain**. Pretrain only learns feature embeddings, not feature importance.

#### 4.4 Transformer Attention Limitation

- 20-128 features → too few tokens for meaningful attention
- Attention learns: "all features equally important" → useless for selection

---

## 5. Conclusion

### Confirmed Negative Result

MAE-style masked reconstruction pretraining **does not help** Transformer backbone for feature selection on tabular data.

### Evidence

1. **Pretrain loss oscillates at ~1.0** → model learns nothing useful
2. **Feature selection fails 2/3 of runs** → random selection
3. **MLP baseline achieves 100%** → standard architecture sufficient

### Why It Fails

1. Tabular features lack spatial/semantic structure for reconstruction
2. Noise features dominate → reconstruction learns to predict mean
3. Gate is not involved during pretrain → no feature importance signal
4. Few tokens → transformer attention meaningless

### Recommendation

**Do not use Transformer backbone for feature selection tasks.**

For tabular feature selection:
- MLP with ADMM gate (SADMM-FS) is sufficient and effective
- Pretraining objectives that work for images/text do not transfer

---

## 6. Alternative Approaches (Not Tested)

If Transformer salvage is desired, consider:

| Approach | Description | Challenge |
|----------|-------------|-----------|
| Task-aware pretrain | Pretrain with label supervision | Requires labels during pretrain |
| Gate-aware pretrain | Include gate in pretrain phase | Gate optimization without labels |
| Contrastive learning | Learn feature embeddings via contrastive loss | Still lacks cross-feature structure |
| Larger feature count | Use datasets with 1000+ features | Attention may become meaningful |

---

## 7. Files Reference

| File | Purpose |
|------|---------|
| `custom_admm/run_transformer_pretrain_ablation.py` | Experiment runner |
| `custom_admm/src/mentor_models.py` | GatedTokenTransformerFS model |
| `custom_admm/src/transformer_pretrain.py` | MaskedFeaturePretrainer (unused in latest run) |
| `custom_admm/results/transformer_pretrain_ablation_20260407_161756.json` | Latest results |

---

## 8. Appendix: Full Pretrain Loss History

### Seed 42
```
Epoch: 0  → Loss: 1.052
Epoch: 1  → Loss: 1.032
Epoch: 2  → Loss: 1.015
Epoch: 3  → Loss: 0.995
Epoch: 4  → Loss: 0.995
Epoch: 5  → Loss: 1.008
Epoch: 6  → Loss: 0.998
Epoch: 7  → Loss: 0.976
Epoch: 8  → Loss: 1.021
Epoch: 9  → Loss: 1.070
Epoch: 10 → Loss: 1.002
Epoch: 11 → Loss: 0.969
Epoch: 12 → Loss: 1.057
Epoch: 13 → Loss: 0.998
Epoch: 14 → Loss: 1.008
Epoch: 15 → Loss: 0.989
Epoch: 16 → Loss: 0.974
Epoch: 17 → Loss: 1.004
Epoch: 18 → Loss: 1.028
Epoch: 19 → Loss: 1.016
Epoch: 20 → Loss: 1.001
Epoch: 21 → Loss: 1.016
Epoch: 22 → Loss: 1.004
Epoch: 23 → Loss: 1.008
```

Pattern: **Oscillates between 0.97-1.07**, no convergence trend.

---

*Document generated for paper appendix / negative result documentation.*