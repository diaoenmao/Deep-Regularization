# Mentor Experiment Report: Pseudocode + Implementation Audit

## Scope

This note is for reporting the mentor-requested experiments from three angles:

1. `gating`
2. `backbone`
3. `training order`

It summarizes:

- how each method is actually implemented in code
- pseudocode for presentation use
- what the current results say
- where the current implementations are weaker or not fully aligned

Primary source files:

- [`custom_admm/src/admm_input_group_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py)
- [`custom_admm/src/mentor_models.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\mentor_models.py)
- [`custom_admm/experiments/mentor_axes_experiments.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\experiments\mentor_axes_experiments.py)
- [`custom_admm/experiments/backbone_tier2_synthetic_full.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\experiments\backbone_tier2_synthetic_full.py)
- [`custom_admm/experiments/training_order_synthetic_full.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\experiments\training_order_synthetic_full.py)

## Evidence Level

This needs to be stated clearly in the report:

| Axis | Evidence level | Source |
|---|---|---|
| `gating` | fixed rerun on 3 tasks | [`mentor_gating_full_20260330_fixed.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\mentor_gating_full_20260330_fixed.json) |
| `backbone + Tier-2` | full synthetic grid | [`backbone_tier2_synthetic_full_20260327.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\backbone_tier2_synthetic_full_20260327.json) |
| `training order` | full synthetic grid | [`training_order_synthetic_full_20260329.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\training_order_synthetic_full_20260329.json) |

So:

- `gating` is now cleaner than before, but still narrower than the full-grid studies
- `backbone` and `training-order` conclusions are still the broadest evidence

## Common Base: What All Our ADMM Variants Actually Do

The common base model is `GatedFeatureSelectionMLP` in [`admm_input_group_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py).

Core mechanism:

```text
Input x in R^m
Learn global feature gate g in R^m
Forward uses x_tilde = x * g_eff
Predict y_hat = backbone(x_tilde)

Train in two phases:
1. Warm-up:
   optimize all parameters with supervised loss only
2. ADMM phase:
   g-step:
      optimize backbone parameters and raw gate parameter with augmented loss
   z-step:
      apply soft-thresholding
      if Ratio Norm is on:
         apply global cubic rescaling
   dual update:
      u <- u + g - z
```

Important implementation facts:

- forward selection score is the learnable `gate` itself if the model has a `gate` parameter
- ADMM penalty weights are feature-adaptive:
  - standard MLP: first-layer column norms
  - expansion model: expansion-weight norms
  - transformer model: token-embedding norms

This means:

- the object used for **ranking** is not always the same object used to compute **adaptive penalty weights**
- that is acceptable, but it should be made explicit

## 1. Gating Axis

### 1.1 Linear / Unbounded Gate

Implemented by:

- `GatedFeatureSelectionMLP(..., bounded_gate=False)`
- used in [`mentor_axes_experiments.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\experiments\mentor_axes_experiments.py)

Pseudocode:

```text
parameters:
    raw gate g in R^m
    MLP weights theta

forward:
    gate_eff = g
    if training and feat_drop > 0:
        randomly zero some entries of gate_eff
        rescale surviving entries
    x_gated = x * gate_eff
    y_hat = MLP(x_gated)

training:
    warm-up supervised training
    ADMM pruning on raw gate g
```

What it means:

- gate and sparsity variable live in the same space
- ADMM pushes the same quantity that the forward pass uses

This is the cleanest implementation.

### 1.2 Sigmoid / Bounded Gate

Implemented by:

- `GatedFeatureSelectionMLP(..., bounded_gate=True)`

Pseudocode:

```text
parameters:
    raw gate g_raw in R^m

forward:
    gate_eff = sigmoid(g_raw)
    if training and feat_drop > 0:
        randomly drop entries in gate_eff
    x_gated = x * gate_eff
    y_hat = MLP(x_gated)

training:
    ADMM still sparsifies g_raw directly
```

### 1.3 Current result

Fixed rerun over:

- `xor_m128`
- `ring_m128`
- `ring+xor_m256`

Overall means:

| Method | Mean best-k | Mean AUC |
|---|---:|---:|
| `linear_gate_mlp` | `0.7361` | `0.6417` |
| `sigmoid_gate_mlp` | `0.7083` | `0.6254` |

### 1.4 Implementation caveat

This was the main issue in the original bounded-gate branch, but it has now been fixed.

Previous problem:

```text
forward used sigmoid(raw_gate)
ADMM sparsified raw_gate directly
```

That made the old quick pilot hard to interpret, because `raw_gate -> 0` implies
`sigmoid(raw_gate) -> 0.5`, not feature shutdown.

Current status:

- ADMM consensus now runs in the **effective gate space**
- bounded-gate reruns should be read using:
  - [gating_fix_rerun_20260330.md](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\analysis\gating_fix_rerun_20260330.md)

Updated reporting implication:

- the original quick pilot is no longer the right evidence source
- use the fixed rerun instead
- after the fix, the unbounded gate is still better, but the conclusion is now methodologically cleaner

## 2. Backbone Axis

### 2.1 Baseline: Gated MLP

Implemented by:

- `GatedFeatureSelectionMLP`
- full synthetic baseline in [`backbone_tier2_synthetic_full.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\experiments\backbone_tier2_synthetic_full.py)

Pseudocode:

```text
input x
apply global scalar gate g
x_gated = x * g
pass x_gated through 2-layer MLP
train with warm-up + ADMM + Ratio Norm
rank features by |g|
```

This is the reference implementation.

### 2.2 Token Transformer Backbone

Implemented by:

- `GatedTokenTransformerFS` in [`mentor_models.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\mentor_models.py)

Pseudocode:

```text
parameters:
    global gate g in R^m
    feature embedding E in R^{m x d}
    cls token
    position embedding
    transformer encoder
    classifier head

forward:
    x_gated = x * g
    token_j = x_gated[j] * E[j]
    sequence = [CLS, token_1, ..., token_m] + position_embedding
    encoded = Transformer(sequence)
    y_hat = classifier(encoded[CLS])

ADMM penalty score:
    use ||E[j]||_2 as per-feature penalty score

feature ranking at evaluation:
    use |g|
```

Important detail:

- the ADMM threshold scaling uses embedding norms
- final feature ranking still uses the gate

So this model is:

- globally gated at the original-feature level
- transformer-encoded after gating

It is not doing attention-based selection directly.

### 2.3 Masked-Pretrained Transformer

Implemented by:

- `GatedTokenTransformerFS` plus `_pretrain_masked_transformer(...)`

Pseudocode:

```text
pretraining:
    randomly mask some input features
    disable ADMM gate during pretraining
    encode masked feature tokens
    reconstruct masked raw features

fine-tuning:
    switch back to gated supervised ADMM training
```

### 2.4 Full synthetic result

Overall averages:

| Method | Mean best-k | Mean best-2k | Mean AUC | Mean AUPRC |
|---|---:|---:|---:|---:|
| `gated_mlp` | `0.6278` | `0.6656` | `0.6605` | `0.6822` |
| `gated_token_transformer` | `0.2479` | `0.3146` | `0.5544` | `0.5693` |
| `gated_token_transformer_pretrained` | `0.2094` | `0.2792` | `0.5481` | `0.5626` |

### 2.5 Implementation caveats

Main caveats:

1. Pretraining is weak and only loosely coupled to the downstream objective

```text
masked reconstruction pretraining
!= feature-selection-aware pretraining
```

The pretraining stage reconstructs masked values, but it does not directly encourage sparse selection or informative gating.

2. Gate is still global and diagonal

This is good for interpretability, but it also means the transformer is not allowed to express richer feature-interaction selection before the gate.

3. The method is expensive relative to the achieved gain

The implementation is working, but the results are decisively worse than the plain gated MLP.

Reporting implication:

- this branch should be reported as a negative result
- not as a near-miss that just needs a little more tuning

## 3. Training-Order Axis

### 3.1 Baseline: Select Then MLP

Implemented by:

- `select_then_mlp`
- same `GatedFeatureSelectionMLP` as the main method

Pseudocode:

```text
input x
apply gate g on raw features
x_gated = x * g
y_hat = MLP(x_gated)
```

### 3.2 Expansion Then Selection Then MLP

Implemented by:

- `ExpandedFeatureSelectionMLP` in [`mentor_models.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\mentor_models.py)

Pseudocode:

```text
parameters:
    expansion matrix W_exp[j] for each original feature j
    global gate g in R^m
    MLP on flattened expanded features

forward:
    for each original feature j:
        token_j = x[j] * W_exp[j]          # local feature expansion
        token_j = token_j * g[j]           # gate entire expanded group
    z = concatenate(token_1, ..., token_m)
    y_hat = MLP(z)
```

This is important:

- selection is still at the **original feature level**
- gate `g[j]` turns on or off the whole expanded block for feature `j`
- this is **not** expanded-dimension-level selection

ADMM penalty score:

```text
score_j = ||W_exp[j]||_2
```

Final ranking at evaluation:

```text
rank by |g_j|
```

### 3.3 Full synthetic result

Overall averages:

| Method | Mean best-k | Mean best-2k | Mean AUC | Mean AUPRC |
|---|---:|---:|---:|---:|
| `select_then_mlp` | `0.6090` | `0.6483` | `0.6628` | `0.6844` |
| `expand4_then_select_then_mlp` | `0.3694` | `0.4181` | `0.6221` | `0.6486` |
| `expand8_then_select_then_mlp` | `0.4184` | `0.4573` | `0.6383` | `0.6621` |
| `expand16_then_select_then_mlp` | `0.4181` | `0.4597` | `0.6537` | `0.6774` |

Most important conclusion:

- no expansion width beats `select_then_mlp` on `best-k`
- `expand16` is the strongest expansion variant
- expansion occasionally helps `AUC`, but not enough to justify replacing the main pipeline

### 3.4 Implementation caveats

1. Expansion increases capacity substantially

The flattened representation size becomes:

```text
m * expand_dim
```

So this branch changes two things at once:

- feature-processing order
- model capacity

That makes attribution less clean.

2. Selection is still original-feature-level

This is conceptually clean, but it means the experiment does **not** test "selection after full learned feature expansion" in the strongest possible sense. It tests:

```text
local per-feature expansion
-> original-feature group gating
-> MLP
```

Reporting implication:

- this is still a valid test of the hypothesis
- but the hypothesis should be described precisely, not as generic "feature expansion then selection"

## 4. Tier-2 Baselines

These were run in the full synthetic backbone/Tier-2 study.

### 4.1 FSNet

Implementation path:

- [`custom_admm/src/core.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\core.py)
- method name `fsnet`

Pseudocode:

```text
set selected-width = min(2k, m)
train FSNet predictor with selected-width
get global feature importance scores from FSNet
rank features by those scores
```

Implementation caveat:

- in this wrapper, FSNet is trained with `2k` selected features, not `k`
- `scores` and `scores2` are the same ranking

So this is a runnable baseline, but not a perfectly symmetric `k`-controlled comparator.

### 4.2 E2E-FS

Implementation path:

- [`custom_admm/src/e2efs_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\e2efs_wrapper.py)

Pseudocode:

```text
split train into fit/validation
fit vendor E2E-FS selector for k features
use selector mask as feature score
predict on train/test using selector's own predictor
fit a second selector for 2k when needed
```

Implementation caveat:

- this is a vendor-backed wrapper using `vendor_pkgs`
- architecture and training loop are method-specific, not matched to ours

That is fine as a Tier-2 baseline, but it is standard-implementation evidence, not controlled-backbone evidence.

### 4.3 CAE

Implementation path:

- [`custom_admm/src/cae_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\cae_wrapper.py)

Pseudocode:

```text
learn k selector rows over m original features
sample Gumbel-softmax selector during training
at evaluation, hard-pick one feature per selector row
selected representation = x @ selector_matrix^T
predict with a small MLP
feature score = sum of selector probabilities over rows
```

Implementation caveat:

- this is **not** the original historical Keras package
- it is a local CAE-style compatible implementation because the old package is incompatible with current Keras

So:

- good as a runnable local proxy
- not strong enough to present as an exact reproduction claim

### 4.4 TabNet

Implementation path:

- [`custom_admm/src/tabnet_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\tabnet_wrapper.py)

Pseudocode:

```text
split train into fit/validation
train TabNetClassifier
use predict_proba for classification outputs
use feature_importances_ as global feature score
rank features by feature_importances_
```

Important clarification:

- TabNet is not transformer-based
- it is an attentive tabular encoder with sequential feature masks

Implementation caveat:

- TabNet does not enforce an explicit `k`-feature selector during training
- we only convert its global feature importance ranking into top-k and top-2k after training

So this is a relevant encoder-style baseline, but not a strict fixed-budget selector.

## 5. Backbone + Tier-2 Full Synthetic Result

Overall averages:

| Method | Mean best-k | Mean best-2k | Mean AUC | Mean AUPRC |
|---|---:|---:|---:|---:|
| `gated_mlp` | `0.6278` | `0.6656` | `0.6605` | `0.6822` |
| `tabnet` | `0.2851` | `0.3476` | `0.5995` | `0.6191` |
| `gated_token_transformer` | `0.2479` | `0.3146` | `0.5544` | `0.5693` |
| `gated_token_transformer_pretrained` | `0.2094` | `0.2792` | `0.5481` | `0.5626` |
| `cae` | `0.1861` | `0.2667` | `0.5236` | `0.5496` |
| `e2efs` | `0.1767` | `0.2559` | `0.5117` | `0.5359` |
| `fsnet` | `0.1667` | `0.2590` | `0.5621` | `0.5867` |

Reporting implication:

- `gated_mlp` remains clearly strongest
- `tabnet` is the strongest Tier-2 baseline
- the transformer branch is a negative result
- `FSNet`, `E2E-FS`, and `CAE` are not competitive on this full synthetic grid

## 6. What Looks Under-Implemented or Misaligned

This is the concise audit list for discussion.

### High priority

1. `CAE` is a local compatible implementation, not the original package

- acceptable for internal benchmarking
- not acceptable as an exact reproduction claim

2. `FSNet` wrapper uses `2k` selected width in training

- reasonable as a practical baseline
- not the cleanest matched-budget comparator

3. `TabNet` is ranked post hoc, not trained as an explicit `k`-selector

- still useful
- but conceptually different from exact-top-k selector methods

### Medium priority

4. Transformer pretraining is weakly coupled to selection

- masked reconstruction may be too generic
- no evidence it helps sparse feature recovery

5. Expansion-before-selection changes both ordering and effective capacity

- if this branch is revisited, a tighter control would hold total parameter budget fixed

## 7. Final Reporting Position

This is the most defensible way to present the mentor experiments.

### Keep

- `select_then_mlp` as the main method
- unbounded global scalar gate as the default gate
- `gated_mlp` as the main backbone
- `tabnet` as the strongest Tier-2 encoder-style baseline

### Keep as ablations / exploratory

- `sigmoid gate`
- `expand -> select -> MLP`
- `gated token transformer`
- `masked-pretrained transformer`

### Do not over-claim

- do not say bounded gating is fundamentally worse without noting the current implementation mismatch
- do not say CAE is an exact literature reproduction
- do not say TabNet is transformer-based
- do not say expansion-before-selection failed in principle; more accurate is:
  - it failed to beat the simpler pipeline under the current implementation and evaluation protocol

## 8. Short Verbal Summary

If this needs to be presented in one minute:

> We tested the mentor directions from three angles. On gating, the current unbounded global gate is better than the sigmoid version, but the sigmoid implementation is not fully aligned with the ADMM sparsity variable, so that result should be read cautiously. On backbone, the plain gated MLP remains clearly stronger than both the transformer variants and all Tier-2 encoder-style baselines; among those baselines, TabNet is the strongest but still well below the main method. On training order, expansion-before-selection does not beat the simpler select-then-MLP pipeline on best-k, although larger expansion widths can occasionally improve AUC on mixed tasks. Overall, the simplest global-gate MLP with ADMM remains the strongest and cleanest implementation.
