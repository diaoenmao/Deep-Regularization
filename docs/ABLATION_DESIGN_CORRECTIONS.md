# Ablation Study Design - Single Variable Principle

**Created**: 2026-04-17
**Purpose**: Ensure each ablation table tests ONLY ONE variable

---

## Current Issues

| Table | Issue | Severity |
|-------|-------|----------|
| 3b (Backbone) | Missing Transformer per-dataset data | HIGH |
| 3c (Iterative) | Strategy differences unclear | MEDIUM |
| 3d (Training Order) | expand changes BOTH order AND capacity | **CRITICAL** |
| 3e (Polynomial) | degree AND selection_mode both change | **CRITICAL** |

---

## Corrected Ablation Design

### Table 3a: Gating (OK - No Change Needed)

**Variable tested**: Gate activation type
**Fixed**: All other hyperparameters

| Config | Gate | Fixed Parameters |
|--------|------|------------------|
| Linear | `g` (unbounded, raw) | backbone=MLP, training=single_pass, C=0.05, epochs=416, warmup=100 |
| Sigmoid | `sigmoid(g)` (bounded, 0-1) | backbone=MLP, training=single_pass, C=0.05, epochs=416, warmup=100 |

**What changes**: Gate computation (raw vs sigmoid)
**What stays fixed**: Everything else

---

### Table 3b: Backbone (Need Data)

**Variable tested**: Backbone architecture
**Fixed**: Gate, training, all hyperparameters

| Config | Backbone | Fixed Parameters |
|--------|----------|------------------|
| MLP | 2-layer MLP, latent=32 | gate=linear, training=single_pass, C=0.05 |
| Transformer | Token Transformer, d=32 | gate=linear, training=single_pass, C=0.05 |

**What changes**: Architecture (MLP vs Transformer encoder)
**What stays fixed**: Gate type, training schedule, hyperparameters

**Missing**: Transformer per-dataset breakdown (need to fill)

---

### Table 3c: Iterative Strategy (Need Clarification)

**Variable tested**: Iterative pruning strategy
**Fixed**: Backbone, gate, all hyperparameters

| Strategy | Description | What Changes |
|----------|-------------|--------------|
| single_pass | One-shot ADMM training | Baseline - no iteration |
| iterative_hard | Multi-phase with hard pruning per phase | Pruning strategy |
| lottery_ticket | iterative_hard + weight reset | Weight initialization |
| gradual_admm | Multi-phase with gradual C increase | C scheduling |

**Issue**: These strategies change DIFFERENT things!
- single_pass vs iterative_hard: changes pruning timing
- iterative_hard vs lottery_ticket: changes weight reset
- gradual_admm vs single_pass: changes C scheduling

**Recommendation**: Split into separate tables:
- 3c-1: Pruning timing (single_pass vs multi_phase)
- 3c-2: Weight reset effect (with_reset vs no_reset)
- 3c-3: C scheduling (gradual vs fixed)

---

### Table 3d: Training Order (INVALID - Redesign Needed)

**Current Problem**: expand4/8/16 changes BOTH:
1. Processing order (expand first vs select first)
2. Model capacity (input dimension changes)

**Corrected Design**:

**Variable tested**: Feature processing order
**Fixed**: Total model capacity (hidden layer size), hyperparameters

| Config | Order | Input Dim | Hidden Dim | Fixed |
|--------|-------|-----------|------------|-------|
| select_then_mlp | Select → MLP | 128 (selected) | 32 | Same total params |
| expand_then_mlp | Expand → Select → MLP | 128 (expanded) | 32 | Same total params |

**What changes**: Processing order ONLY
**What stays fixed**: Model capacity, hyperparameters

**Note**: Current expand experiments should be removed or renamed to "Model Capacity Ablation"

---

### Table 3e: Polynomial (INVALID - Redesign Needed)

**Current Problem**: Tests TWO variables:
1. Polynomial degree (1 vs 2)
2. Selection mode (group vs expanded)

**Corrected Design**: Split into TWO tables

**3e-1: Polynomial Degree**
| Config | Degree | Selection Mode | Fixed |
|--------|--------|----------------|-------|
| degree=1, group | 1 | group | Same training, same hyperparams |
| degree=2, group | 2 | group | Same training, same hyperparams |

**What changes**: Polynomial degree ONLY
**What stays fixed**: Selection mode=group, all hyperparameters

**3e-2: Selection Mode**
| Config | Degree | Selection Mode | Fixed |
|--------|--------|----------------|-------|
| degree=2, group | 2 | group (select original features) | Same degree |
| degree=2, expanded | 2 | expanded (select expanded features) | Same degree |

**What changes**: Selection granularity ONLY
**What stays fixed**: Degree=2, all hyperparameters

---

### Table 4: Gradual Pruning (OK - Well Designed)

**Variables tested**: Two variables tested in factorial design
- Pruning mode: soft (mask) vs hard (delete)
- Re-weighting: rw vs no_rw

This is actually GOOD - tests two variables independently.

| Variant | Pruning | Re-weight | What Tested |
|---------|---------|-----------|-------------|
| soft_no_rw | soft | no | Baseline |
| soft+rw | soft | yes | Re-weight effect on soft |
| hard_no_rw | hard | no | Pruning mode effect |
| hard+rw | hard | yes | Re-weight effect on hard |

**Comparison pairs**:
- soft_no_rw vs soft+rw: Tests re-weighting (pruning fixed)
- soft_no_rw vs hard_no_rw: Tests pruning mode (re-weight fixed)
- soft+rw vs hard+rw: Tests pruning mode with re-weight

---

## Recommended Ablation Structure

### Reorganized Tables

| Table | Variable | Configurations | Status |
|-------|----------|----------------|--------|
| 3a | Gate type | Linear vs Sigmoid | OK |
| 3b | Backbone | MLP vs Transformer | Need data |
| 3c-1 | Pruning timing | single_pass vs multi_phase | Need redesign |
| 3c-2 | Weight reset | with_reset vs no_reset | Need new experiment |
| 3c-3 | C scheduling | gradual vs fixed | OK (gradual_admm) |
| 3d | Processing order | select_first vs expand_first | Need redesign |
| 3d-2 | Model capacity | Different hidden sizes | Separate table |
| 3e-1 | Polynomial degree | degree=1 vs degree=2 | Need redesign |
| 3e-2 | Selection mode | group vs expanded | Need redesign |
| 4a | Pruning mode | soft vs hard | OK (split from 4b) |
| 4b | Re-weighting | rw vs no_rw | OK |

---

## Explanations for Each Ablation

### 3a: Why Gate Type Matters?
- Linear gate: unbounded, can go negative, allows true "off" state
- Sigmoid gate: bounded (0-1), always "somewhat on", harder to achieve sparsity

### 3b: Why Backbone Matters?
- MLP: Simple architecture, direct feature interaction
- Transformer: Attention-based, but lacks spatial structure for tabular features

### 3c: Why Iterative Matters?
- single_pass: One-shot learning, limited refinement
- iterative: Multi-phase allows gradual feature elimination
- lottery_ticket: Tests whether weight reset helps (hypothesis: NO)

### 3d: Why Order Matters?
- select_first: Choose features first, then train predictor
- expand_first: Expand features first, then select from expanded space

### 3e: Why Polynomial Matters?
- degree=1: Linear features only, cannot capture Ring boundary
- degree=2: Polynomial features capture Ring (x^2 + y^2 = r^2)

### 4: Why Pruning Mode Matters?
- soft: Mask weak gates, keep all weights, let gates naturally decay
- hard: Delete features, reduce dimension, lose learned weights
- re-weight: Attempt to maintain gate energy, but hurts performance