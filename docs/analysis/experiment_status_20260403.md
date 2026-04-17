# Experiment Status Summary

**Generated**: 2026-04-03

## Completed Work

### Phase 1: Fairness Control ✅

1. **Matched neural-only synthetic benchmark harness** - COMPLETED
   - Results: `matched_neural_synthetic_20260326_131516.json`
   - Methods tested: SADMM-FS, STG, CancelOut, DeepPINK
   - Match level: full_match for CancelOut/DeepPINK, backbone_only for STG

2. **STG reproducibility** - RESOLVED
   - `stg_wrapper.py` exists and works correctly
   - Uses official `stg` package

3. **Fairness protocol documentation** - COMPLETED
   - `analysis/fairness_protocol_20260403.md`

### Baseline Implementations ✅

| Method | Status | Implementation |
|--------|--------|----------------|
| SADMM-FS | ✅ Complete | `admm_input_group_wrapper.py` |
| STG | ✅ Complete | `stg_wrapper.py` |
| CancelOut | ✅ Complete | `cancelout.py` |
| DeepPINK | ✅ Complete | `deeppink.py` |
| LassoNet | ✅ Complete | pip package `lassonet` |
| FSNet | ✅ Complete | `fsnet.py` |
| E2E-FS | ✅ Complete | `e2efs_wrapper.py` |
| CAE | ✅ Complete | `cae_wrapper.py` |
| TabNet | ✅ Complete | `tabnet_wrapper.py` |
| Transformer backbone | ✅ Complete | `mentor_models.py` (negative result) |

### Experiment Results ✅

| Experiment | Status | Results File |
|------------|--------|--------------|
| Gating (linear vs sigmoid) | ✅ Complete | `mentor_gating_full_20260330_fixed.json` |
| Backbone (MLP vs transformer) | ✅ Complete | `backbone_tier2_synthetic_full_20260327.json` |
| Tier-2 baselines | ✅ Complete | `tier2_rerun_fixed_20260402.json` |
| Training order | ✅ Complete | `training_order_synthetic_full_20260329.json` |
| Matched neural | ✅ Complete | `matched_neural_synthetic_20260326_131516.json` |
| Real-world datasets | ✅ Complete | `external-data/*.json` |
| Transformer penalty ablation | ✅ Complete | `transformer_penalty_ablation.json` |

---

## Remaining Work (from TODO_TKDE_20260326.md)

### Phase 2: Method Simplification / Redesign

#### 2.1 Column-normalized scalar gate variant
**Status**: NOT STARTED
**Priority**: Medium

Goal: Implement a column-normalized gate to reduce scale ambiguity between gate and first-layer weights.

Deliverables:
- [ ] New model variant in `custom_admm/src/`
- [ ] Training integration
- [ ] Ablation against current SADMM-FS

#### 2.2 Feature dropout ablation
**Status**: NOT STARTED
**Priority**: High

Goal: Determine if dropout is necessary with the normalized gate variant.

Deliverables:
- [ ] Ablation table: with dropout / without dropout
- [ ] Under controlled setup

### Phase 3: Theory Support

#### 3.1 Reframe method theory
**Status**: NOT STARTED
**Priority**: High

Core theory should focus on:
- Diagonal/global gate variable
- Linearized ADMM splitting
- Proximal sparsification
- Ratio norm vs L1
- Identifiability from column normalization

#### 3.2 Source-backed theory note
**Status**: NOT STARTED
**Priority**: Medium

Deliverable: Markdown note under `analysis/` summarizing the theory story.

### Phase 4: Reruns
**Status**: BLOCKED by Phase 2 and 3

### Phase 5: Paper Revision
**Status**: BLOCKED by Phase 2-4

---

## Key Findings

### Main Result
SADMM-FS (gated_mlp) with linear gate consistently outperforms all baselines on:
- Synthetic feature-selection benchmarks (best-k recovery)
- Real-world high-dimensional datasets (AUROC)

### Baseline Rankings (by mean best-k on synthetic)

| Rank | Method | Mean best-k | Match Level |
|------|--------|-------------|-------------|
| 1 | SADMM-FS | 0.8125 | method_specific |
| 2 | STG | 0.625 | backbone_only |
| 2 | CancelOut | 0.625 | full_match |
| 4 | TabNet | 0.2851 | standard_impl |
| 5 | CAE | 0.1861 | standard_impl |
| 6 | E2E-FS | 0.1767 | standard_impl |
| 7 | FSNet | 0.1667 | standard_impl |
| 8 | DeepPINK | 0.09375 | full_match |

### Negative Results

1. **Transformer backbone**: Fails completely on synthetic FS benchmarks (best-k ≈ 0)
2. **Sigmoid gate**: Consistently underperforms linear gate
3. **Expansion before selection**: Hurts best-k recovery
4. **DeepPINK**: Not suitable for neural FS benchmarks (designed for knockoff statistics)

---

## Next Steps

1. **Implement column-normalized gate variant** (Phase 2.1)
2. **Run feature dropout ablation** (Phase 2.2)
3. **Write theory note** (Phase 3.2)
4. **Regenerate paper-critical results** after Phase 2 decisions