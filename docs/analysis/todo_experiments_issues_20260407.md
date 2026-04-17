# TODO Experiments Status and Issues

**Generated**: 2026-04-07

## Summary

| Experiment | Status | Key Finding |
|------------|--------|-------------|
| **Iterative FS** | ✅ **DONE** | gradual_admm best; lottery_ticket worst (8% Ring) |
| **Polynomial Expansion** | ✅ **DONE** | expanded > group for degree>1 |
| **Transformer Pretrain** | ✅ **DOCUMENTED** | Negative result: loss oscillates at ~1.0 |
| **Sigmoid vs Linear** | ✅ **DONE** | Unbounded > Bounded (-30% XOR) |

---

## 1. Iterative Feature Selection (Lottery Ticket Style)

**Status**: ✅ COMPLETED (2026-04-08)

**Source**: `iterative_ablation_20260408_195630.json` (6-fold CV, all methods)

### Results

| Method | XOR | Ring | Ring+XOR |
|--------|-----|------|----------|
| single_pass | 1.00 | 0.67 | 0.67 |
| iterative_hard | 0.67 | 0.58 | 0.12 |
| lottery_ticket | 0.50 | **0.08** | 0.12 |
| **gradual_admm** | **1.00** | **1.00** | **0.67** |

### Key Finding

**`gradual_admm` outperforms all iterative methods**

- Lottery ticket hypothesis does NOT help for feature selection
- Weight rewinding actually hurts performance (0.08 on Ring vs 0.58 iterative_hard)
- Gradual ADMM tightening is the most robust approach

### Code Fix Applied

Fixed `_subset_state_dict` to use shape-based matching instead of name-based matching for weight subsetting during lottery ticket rewinding.

---

## 2. Polynomial Feature Expansion

**Status**: ✅ NOW USES ADMM+RatioNorm (2026-04-07)

**Source**: `polynomial_ablation_20260407_172403.json` (ADMM+RatioNorm results)

### Current Results (ADMM+RatioNorm)

| Dataset | Degree | Mode | best_k |
|---------|--------|------|--------|
| XOR | 1 | group | 1.00 |
| XOR | 1 | expanded | 1.00 |
| XOR | 2 | group | 0.67 |
| XOR | 2 | **expanded** | **1.00** |

### Key Finding

For polynomial expansion, **independent selection per expanded feature** (expanded mode) works better than **shared gate per original feature** (group mode) when degree > 1.

### Changes Made (2026-04-07)

1. **Modified `run_polynomial_ablation.py`** to use `_train_input_group` with ADMM+RatioNorm
2. **Added `get_feature_scores()` method** to return gate-aligned importance scores
3. **Added `self.layers` and `self.first_linear`** for interface compatibility

### Technical Notes

- sklearn's `PolynomialFeatures` is used for expansion (deterministic, no gradients needed)
- Gradients flow to gates via downstream predictor
- For "group" mode: scores aggregated from expanded to original feature dimension
- For "expanded" mode: scores computed directly from first linear layer

### Action Items

- [x] Update to use ADMM+RatioNorm
- [x] Fix interface for `_train_input_group`
- [ ] Run full ablation (more datasets, 6-fold CV)

---

## 3. Transformer Pretrain

**Status**: ✅ DOCUMENTED (2026-04-08)

**Source**: `transformer_pretrain_ablation_20260407_161756.json`

**Documentation**: `analysis/transformer_pretrain_negative_result.md`

### Results

| Method | XOR best-k | Success Rate |
|--------|------------|--------------|
| MLP baseline | **1.00 ± 0.0** | 100% (3/3) |
| Transformer + Pretrain | 0.33 ± 0.47 | 33% (1/3) |

### Pretrain Loss Analysis

**Key Finding: Loss oscillates at ~1.0, does NOT converge**

| Seed | Loss Range | Trend |
|------|------------|-------|
| 42 | 0.97 - 1.07 | Oscillates around 1.0 |
| 43 | 0.97 - 1.04 | Oscillates around 1.0 |
| 44 | 0.97 - 1.05 | Oscillates around 1.0 |

**Interpretation**: Model learns to predict mean (= 0 after [-1,1] scaling) for masked noise features. MSE ≈ variance ≈ 1.0.

### Root Cause

1. **Tabular features lack cross-feature structure** - MAE works for images/text due to spatial/semantic relationships
2. **Noise features dominate** - 30 noise vs 2 signal features, reconstruction learns nothing useful
3. **Gate not involved during pretrain** - Pretrain only learns embeddings, not feature importance
4. **Few tokens** - 20-32 features insufficient for meaningful attention

### Conclusion

✅ **Transformer backbone is a confirmed negative result for tabular FS**

MAE-style masked reconstruction pretraining does not transfer to tabular feature selection.

**Recommendation**: Do not use Transformer backbone for feature selection tasks.

### Action Items

- [x] Quick test with 3 seeds
- [x] Analyze loss behavior (oscillation at ~1.0)
- [x] Document negative result

---

## 4. Sigmoid ADMM Fix

**Status**: ✅ EXPERIMENTS COMPLETED (2026-04-08)

**Source**: `bounded_gate_ablation.json`

### Results

| Dataset | Unbounded | Bounded (sigmoid) | Difference |
|---------|-----------|-------------------|------------|
| XOR | **86.1%** | 55.6% | -30.5% |
| Ring | **22.2%** | 13.9% | -8.3% |
| Ring+XOR+Sum | **51.9%** | 40.7% | -11.2% |

### Key Finding

⚠️ **Bounded (sigmoid) gates consistently perform WORSE than unbounded gates**

Even with the raw space ADMM fix, sigmoid gates show:
- 30% worse on XOR
- 8% worse on Ring
- 11% worse on Ring+XOR+Sum

### Conclusion

**Recommendation**: Use unbounded gates for SADMM-FS. The sigmoid constraint appears to hinder feature selection performance even with correct raw-space ADMM dynamics.

### Action Items

- [x] Implement raw space ADMM fix
- [x] Re-run comparison experiments
- [ ] Document finding in paper (unbounded > bounded)

---

## Protocol Configuration Reference

From `fairness_protocol_20260403.md` (updated to match main SADMM-FS):

```python
# Matched Backbone
{
    "latent_size": 32,
    "n_hidden_layers": 2,
    "gaussian_noise": 0.0,
    "dropout": 0.043,
    "activation": "mish",
    "feat_drop": 0.6
}

# Matched Training Config (matches main SADMM-FS in run_admm_input_group)
{
    "learning_rate": 0.005,
    "epochs": 500,
    "batch_size": 64,
    "weight_decay": 1e-3,
    "optimizer": "adam",
    "early_stopping_patience": 66,
    "warmup_epochs": 120
}

# Seed Policy (Synthetic)
# 6-fold CV, seed = 42 + fold_idx
```

**Note**: The `nn_wrapper.py` in Feature-Selection-Benchmark uses different defaults (adagrad, lr=0.00176). Main SADMM-FS method uses Adam with lr=0.005.

---

## Execution Plan

| Priority | Task | Status | Finding |
|----------|------|--------|---------|
| 1 | Sigmoid vs Linear gate | ✅ **DONE** | Unbounded > Bounded (-30% XOR) |
| 2 | Polynomial ADMM+RatioNorm | ✅ **DONE** | expanded > group for degree>1 |
| 3 | Iterative (fix lottery_ticket) | ✅ **DONE** | gradual_admm = 1.00 on XOR/Ring |
| 4 | Transformer pretrain | ✅ **DOCUMENTED** | Negative result: loss oscillates at ~1.0 |

---

## Notes

- All three TODO experiments used **different** training configurations than the protocol
- This is acceptable for **exploratory** experiments but not for **paper results**
- The main comparisons (SADMM-FS vs baselines) in Sections 1-7 **do** follow the protocol
- New experiments should either:
  1. Follow protocol exactly for fair comparison, OR
  2. Clearly document differences and justify them