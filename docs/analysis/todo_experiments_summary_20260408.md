# TODO Experiments Summary Report

**Date**: 2026-04-08
**Status**: All experiments completed

---

## Overview

This report summarizes 4 ablation experiments conducted to validate design decisions for SADMM-FS (Sparse ADMM Feature Selection).

| Experiment | Question | Result |
|------------|----------|--------|
| Iterative FS | Does lottery ticket hypothesis help? | **No** - gradual ADMM best |
| Polynomial Expansion | How to select polynomial features? | **Independent per expanded feature** |
| Transformer Pretrain | Can pretraining salvage Transformer? | **No** - confirmed negative result |
| Sigmoid vs Linear | Should gates be bounded? | **No** - unbounded outperforms |

---

## 1. Iterative Feature Selection

### Question
Does iterative pruning with weight rewinding (Lottery Ticket Hypothesis) improve feature selection?

### Methods Tested

| Method | Description |
|--------|-------------|
| single_pass | Standard ADMM training, single run |
| iterative_hard | Iterative pruning, retrain from current weights |
| lottery_ticket | Iterative pruning with weight rewinding to init |
| gradual_admm | Gradually increase ADMM penalty C |

### Results (6-fold CV)

| Method | XOR | Ring | Ring+XOR |
|--------|-----|------|----------|
| single_pass | 1.00 | 0.67 | 0.67 |
| iterative_hard | 0.67 | 0.58 | 0.12 |
| lottery_ticket | 0.50 | **0.08** | 0.12 |
| **gradual_admm** | **1.00** | **1.00** | **0.67** |

### Conclusion

**Lottery Ticket Hypothesis does NOT apply to feature selection.**

- Weight rewinding hurts performance (Ring: 0.08 vs 0.58 without rewinding)
- Gradual ADMM tightening is the most robust approach
- Single-pass ADMM already achieves optimal on XOR

**Recommendation**: Use gradual_admm for iterative refinement when needed.

---

## 2. Polynomial Feature Expansion

### Question
When using polynomial features, should we assign one gate per original feature (group) or one gate per expanded feature (expanded)?

### Setup

- PolynomialFeatures from sklearn (degree 1-2)
- ADMM+RatioNorm for feature selection
- XOR dataset, 3 seeds

### Results

| Degree | Mode | best_k | Interpretation |
|--------|------|--------|----------------|
| 1 | group | 1.00 | Both modes equivalent (no expansion) |
| 1 | expanded | 1.00 | Both modes equivalent |
| 2 | group | 0.67 | Shared gate dilutes signal |
| 2 | **expanded** | **1.00** | Independent selection works |

### Conclusion

**For polynomial features, use independent gate per expanded feature.**

When degree > 1:
- Group mode: one gate controls multiple polynomial terms → signal dilution
- Expanded mode: each polynomial term has independent gate → precise selection

**Recommendation**: Use "expanded" mode for polynomial feature selection.

---

## 3. Transformer Pretrain

### Question
Can MAE-style masked reconstruction pretraining salvage Transformer backbone for tabular feature selection?

### Setup

- GatedTokenTransformerFS (d_model=32, 2 layers, 4 heads)
- Mask 15% features, reconstruct via MSE
- 24 pretrain epochs, then ADMM fine-tuning
- XOR dataset, 3 seeds

### Results

| Method | best_k | Success Rate |
|--------|--------|--------------|
| MLP Baseline (SADMM-FS) | **1.00 ± 0.0** | 100% |
| Transformer + Pretrain | 0.33 ± 0.47 | 33% |

### Pretrain Loss Behavior

Loss oscillates around 1.0 throughout training:
- Starts at ~1.05, ends at ~1.00
- **Does NOT converge** - model learns to predict mean value
- Range: 0.97 - 1.07 (high variance)

### Root Cause Analysis

1. **No cross-feature structure**: Tabular features are independent; masking one cannot be reconstructed from others
2. **Noise dominance**: 30 noise features vs 2 signal → reconstruction learns nothing useful
3. **Gate not involved**: Pretrain only learns embeddings, not feature importance
4. **Insufficient tokens**: 20-32 features too few for meaningful attention

### Conclusion

**Confirmed negative result. MAE-style pretraining does NOT transfer to tabular FS.**

Transformer backbone is fundamentally unsuitable for tabular feature selection due to:
- Lack of spatial/semantic structure in features
- Reconstruction objective teaches nothing about feature importance

**Recommendation**: Use MLP backbone (SADMM-FS) for tabular feature selection.

---

## 4. Sigmoid vs Linear Gate

### Question
Should ADMM gates be bounded (sigmoid) or unbounded (linear)?

### Setup

- Compare sigmoid(gate) vs raw gate values
- Raw-space ADMM fix applied for sigmoid gates
- XOR, Ring, Ring+XOR+Sum datasets
- 6-fold CV

### Results

| Dataset | Unbounded | Bounded | Difference |
|---------|-----------|---------|------------|
| XOR | **86.1%** | 55.6% | **-30.5%** |
| Ring | **22.2%** | 13.9% | **-8.3%** |
| Ring+XOR+Sum | **51.9%** | 40.7% | **-11.2%** |

### Analysis

Even with correct raw-space ADMM dynamics for sigmoid gates:
- Bounded gates consistently underperform
- Worst on XOR: 30% relative drop
- Sigmoid constraint limits gate expressiveness

### Conclusion

**Unbounded gates outperform bounded gates by 8-30%.**

Bounded sigmoid gates:
- Constrain gate values to [0, 1]
- Limit gradient flow during ADMM optimization
- Reduce ability to express strong feature importance

**Recommendation**: Use unbounded gates for SADMM-FS.

---

## Summary Table

| Experiment | Design Question | Answer | Key Evidence |
|------------|-----------------|--------|--------------|
| Iterative FS | Lottery ticket helpful? | **No** | gradual_admm: 1.00 vs lottery: 0.08 (Ring) |
| Polynomial | Group vs expanded? | **Expanded** | degree=2: expanded 1.00 vs group 0.67 |
| Transformer | Pretrain salvage? | **No** | MLP 100% vs Transformer 33%; loss oscillates |
| Sigmoid | Bounded better? | **No** | XOR: unbounded 86% vs bounded 56% |

---

## Design Recommendations for SADMM-FS

Based on these experiments, the recommended configuration is:

1. **Backbone**: MLP (not Transformer)
2. **Gate**: Unbounded (not sigmoid)
3. **Iterative**: Use gradual_admm if refinement needed (not lottery_ticket)
4. **Polynomial**: Use expanded mode (independent per expanded feature)

---

## Files Reference

| Experiment | Result File | Code |
|------------|-------------|------|
| Iterative | `iterative_ablation_20260408_195630.json` | `src/iterative_run.py` |
| Polynomial | `polynomial_ablation_20260407_172403.json` | `src/polynomial_expansion.py` |
| Transformer | `transformer_pretrain_ablation_20260407_161756.json` | `run_transformer_pretrain_ablation.py` |
| Sigmoid | `bounded_gate_ablation.json` | `run_bounded_gate_ablation.py` |

---

*Report generated for paper documentation.*