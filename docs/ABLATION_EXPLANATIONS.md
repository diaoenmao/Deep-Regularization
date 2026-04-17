# Ablation Study - Complete Explanation

**Purpose**: Each table tests ONE variable, explains what changes

---

## Table 3a: Gate Type

**Question**: Should gate be bounded (sigmoid) or unbounded (linear)?

**What changes**: Gate computation
- Linear: `g` is raw scalar, can be any value (negative, zero, positive)
- Sigmoid: `sigmoid(g)` forces gate to 0-1 range, always "partially on"

**What stays fixed**: MLP backbone, single-pass training, C=0.05, epochs=416, warmup=100

**Results**:
| Gate | XOR | Ring | Ring+XOR | Mean |
|------|-----|------|----------|------|
| Linear | 1.00 | 0.67 | 0.54 | **0.74** |
| Sigmoid | 1.00 | 0.58 | 0.54 | 0.71 |

**Explanation**: Linear gate allows true "off" state (g=0), better for sparsity

---

## Table 3b: Backbone Architecture

**Question**: MLP or Transformer backbone for feature selection?

**What changes**: Architecture type
- MLP: 2-layer feedforward, direct feature interaction
- Transformer: Token embedding + attention encoder

**What stays fixed**: Gate=linear, single-pass training, C=0.05, latent=32

**Results**:
| Backbone | XOR | Ring | Ring+XOR | Mean best-k | Mean AUC |
|----------|-----|------|----------|-------------|----------|
| MLP | 1.00 | 0.50 | 0.62 | **0.63** | 0.66 |
| Transformer | ? | ? | ? | 0.25 | 0.55 |

**Explanation**: Tabular data lacks spatial structure, attention doesn't help

**Missing**: Transformer per-dataset breakdown (need data)

---

## Table 3c: Iterative Training Strategy

**Question**: How should iterative feature selection be done?

**What changes**: Training/pruning strategy
- single_pass: Baseline, one-shot ADMM (no iteration)
- iterative_hard: Multi-phase with hard feature deletion per phase
- lottery_ticket: iterative_hard + weight reset after each phase
- gradual_admm: Multi-phase with gradual C increase (no hard deletion)

**What stays fixed**: MLP backbone, gate=linear, epochs per phase

**Results**:
| Strategy | What Changes | XOR | Ring | Ring+XOR |
|----------|--------------|-----|------|----------|
| single_pass | None (baseline) | 1.00 | 0.67 | 0.67 |
| iterative_hard | +Hard prune per phase | 0.67 | 0.58 | 0.13 |
| lottery_ticket | +Weight reset | 0.50 | 0.08 | 0.13 |
| gradual_admm | +Gradual C (no prune) | 1.00 | **1.00** | 0.67 |

**Explanation**:
- Hard pruning hurts (features deleted, weights lost)
- Weight reset hurts more (lottery ticket hypothesis fails for FS)
- Gradual C increase helps (gentler sparsification)

---

## Table 3d: REMOVED (Invalid)

**Problem**: Original expand experiments changed TWO variables:
1. Processing order (expand first vs select first)
2. Model capacity (input dimension changes)

**Why removed**: Cannot attribute performance to "order" when capacity also changes

---

## Table 3e-1: Polynomial Degree

**Question**: Does polynomial feature expansion help Ring detection?

**What changes**: Polynomial degree of input features
- degree=1: Original features [x1, x2, ...]
- degree=2: Expanded [x1, x2, x1², x2², x1*x2, ...]

**What stays fixed**: MLP backbone, group selection mode, training

**Results**:
| Degree | XOR | Ring | Ring+XOR | What It Adds |
|--------|-----|------|----------|--------------|
| 1 | 1.00 | 0.10 | 0.55 | Linear features |
| 2 | ? | **0.40** | **0.75** | +x², xy (Ring boundary) |

**Explanation**: Ring boundary is circular (x²+y²=r²), degree=2 captures this

---

## Table 3e-2: Selection Mode

**Question**: Should we select from original or expanded features?

**What changes**: Selection granularity
- group: Select original features (each selection turns off entire expansion group)
- expanded: Select individual expanded features (finer control)

**What stays fixed**: Degree=2, MLP backbone, training

**Results**:
| Mode | XOR | Ring | Ring+XOR | Meaning |
|------|-----|------|----------|---------|
| group | ? | ? | ? | Select x1 → turn off x1, x1², x1*x2 |
| expanded | 1.00 | ? | ? | Select x1² alone, keep x1 |

**Explanation**: Group selection is cleaner for interpretation

---

## Table 4a: Pruning Mode

**Question**: Soft mask or hard deletion during gradual pruning?

**What changes**: Pruning operation
- soft: Mask weak gates (g=0), keep weights, let gates decay naturally
- hard: Delete features from model, reduce input dimension

**What stays fixed**: Gradual training, re-weighting=disabled

**Results**:
| Mode | XOR | Ring | Ring+XOR | What Happens |
|------|-----|------|----------|--------------|
| soft | 1.00 | **1.00** | **1.00** | Gate=0, weights intact |
| hard | 1.00 | 0.50 | 0.50 | Feature removed, weights lost |

**Explanation**: Soft pruning preserves learned weights, allows recovery

---

## Table 4b: Re-weighting

**Question**: Should surviving gates be re-weighted after pruning?

**What changes**: Gate scaling after pruning
- no_rw: No re-weighting, gates stay as-is
- rw: Scale surviving gates to maintain total gate energy

**What stays fixed**: Gradual training, soft pruning

**Results**:
| Re-weight | XOR | Ring | Ring+XOR | What Happens |
|-----------|-----|------|----------|--------------|
| no_rw | 1.00 | **1.00** | **1.00** | Gates naturally decay |
| rw | 1.00 | 0.17 | 1.00 | Gates pushed back up |

**Explanation**: Re-weighting pushes gates toward 1, blocking further pruning