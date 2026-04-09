# Mentor Experiment Integrated Report

## Scope

This file consolidates the mentor-requested experiments into one report:

1. `gating`
2. `backbone`
3. `training method`

For each branch, it records:

- the idea
- the actual implementation in code
- the resulting evidence
- the current conclusion

Primary source files:

- [`custom_admm/src/admm_input_group_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py)
- [`custom_admm/src/mentor_models.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\mentor_models.py)
- [`custom_admm/experiments/mentor_axes_experiments.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\experiments\mentor_axes_experiments.py)
- [`custom_admm/experiments/backbone_tier2_synthetic_full.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\experiments\backbone_tier2_synthetic_full.py)
- [`custom_admm/experiments/training_order_synthetic_full.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\experiments\training_order_synthetic_full.py)

Primary result files:

- [`mentor_gating_full_20260330_fixed.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\mentor_gating_full_20260330_fixed.json)
- [`backbone_tier2_synthetic_full_20260327.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\backbone_tier2_synthetic_full_20260327.json)
- [`training_order_synthetic_full_20260329.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\training_order_synthetic_full_20260329.json)

## Executive Summary

- The strongest current design is still the plain `gated_mlp`: global scalar gate, ADMM, Ratio Norm, `feat_drop=0.6`.
- After fixing the bounded-gate implementation mismatch, the `linear/unbounded gate` still beats the `sigmoid/bounded gate`.
- `transformer backbone + ADMM gate` does not work well on the synthetic feature-selection grid, and masked pretraining does not rescue it.
- Among Tier-2 encoder-style baselines, `TabNet` is the strongest, but it is still well below `gated_mlp`.
- `feature expansion -> feature selection -> MLP` does not beat the simpler `select_then_mlp` pipeline on the main metric `best-k`.

So the current recommendation is:

- keep `gated_mlp` as the main method
- keep `sigmoid gate`, `transformer backbone`, and `expand -> select -> MLP` as internal ablations
- keep `TabNet` as the most relevant Tier-2 baseline

## Common Base Method

All our method variants still inherit the same basic structure:

```text
input x in R^m
learn global feature gate g in R^m
compute effective gated input x_tilde = x * g_eff
predict y_hat with a backbone network

training:
1. warm-up supervised optimization
2. ADMM phase:
   a. optimize model parameters and gate with augmented loss
   b. proximal sparsity step on gate consensus variable
   c. dual update
```

Core code path:

- [`GatedFeatureSelectionMLP`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py)
- [`_train_input_group(...)`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py)

This matters because the mentor experiments are not random unrelated models. They are variations on:

- the gate parameterization
- the backbone after gating
- the order of expansion and selection

## Evidence Levels

| Branch | Evidence level | Comment |
|---|---|---|
| `gating` | focused rerun on 3 tasks | good enough to decide default gate, but narrower than a full grid |
| `backbone` | full synthetic grid | strong evidence |
| `Tier-2 baselines` | full synthetic grid | strong evidence |
| `training method` | full synthetic grid | strong evidence |

## 1. Gating

### 1.1 Idea

Question:

> Should the global gate be unbounded and linear, or should it be bounded through a sigmoid?

Motivation:

- bounded gates look more interpretable
- but they may also reduce optimization flexibility

### 1.2 Implementation

#### A. Linear / Unbounded Gate

Implementation:

- `GatedFeatureSelectionMLP(..., bounded_gate=False)`

Pseudocode:

```text
parameters:
    raw gate g in R^m
    MLP weights theta

forward:
    gate_eff = g
    if training and feat_drop > 0:
        randomly drop entries of gate_eff
        rescale survivors
    x_gated = x * gate_eff
    y_hat = MLP(x_gated)

training:
    warm-up
    ADMM on the same gate quantity used by the forward pass
```

#### B. Sigmoid / Bounded Gate

Implementation:

- `GatedFeatureSelectionMLP(..., bounded_gate=True)`

Pseudocode:

```text
parameters:
    raw gate g_raw in R^m

forward:
    gate_eff = sigmoid(g_raw)
    if training and feat_drop > 0:
        randomly drop entries of gate_eff
    x_gated = x * gate_eff
    y_hat = MLP(x_gated)

training:
    ADMM consensus is now applied in effective-gate space
```

Important fix:

- earlier, ADMM sparsified `raw_gate` directly
- now, ADMM acts on the same effective gate used in forward computation

Fix summary:

- [`gating_fix_rerun_20260330.md`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\analysis\gating_fix_rerun_20260330.md)

### 1.3 Result

Source:

- [`mentor_gating_full_20260330_fixed.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\mentor_gating_full_20260330_fixed.json)

Protocol:

- tasks:
  - `xor_m128`
  - `ring_m128`
  - `ring+xor_m256`
- `6` folds
- `240` epochs
- `60` warmup epochs

Per-task results:

| Task | Linear gate best-k / AUC | Sigmoid gate best-k / AUC |
|---|---:|---:|
| `xor_m128` | `1.0000 / 0.8532` | `1.0000 / 0.8448` |
| `ring_m128` | `0.6667 / 0.5126` | `0.5833 / 0.4905` |
| `ring+xor_m256` | `0.5417 / 0.5592` | `0.5417 / 0.5408` |

Overall means:

| Method | Mean best-k | Mean AUC |
|---|---:|---:|
| `linear_gate_mlp` | `0.7361` | `0.6417` |
| `sigmoid_gate_mlp` | `0.7083` | `0.6254` |

### 1.4 Conclusion

- Conclusion still favors the unbounded linear gate.
- The margin is not huge, but it is consistent.

Current decision:

- use `linear/unbounded gate` as the default
- keep `sigmoid/bounded gate` as ablation only

## 2. Backbone

### 2.1 Idea

Question:

> Can we keep the same global gate + ADMM framework but replace the MLP backbone with a richer encoder, especially a transformer-style backbone?

Motivation:

- richer contextual encoding may help nonlinear interactions
- pretraining might help the model learn structure before sparsification

### 2.2 Implementation

#### A. Baseline: Gated MLP

Implementation:

- `GatedFeatureSelectionMLP`

Pseudocode:

```text
input x
apply global scalar gate g
x_gated = x * g
y_hat = MLP(x_gated)
rank features by |g|
```

#### B. Token Transformer Backbone

Implementation:

- `GatedTokenTransformerFS` in [`mentor_models.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\mentor_models.py)

Pseudocode:

```text
parameters:
    global gate g in R^m
    feature embedding E in R^{m x d}
    CLS token
    positional embedding
    transformer encoder
    classifier head

forward:
    x_gated = x * g
    token_j = x_gated[j] * E[j]
    sequence = [CLS, token_1, ..., token_m] + position embedding
    encoded = Transformer(sequence)
    y_hat = classifier(encoded[CLS])

ADMM score:
    score_j = |g_j|  (gate magnitude, not embedding norm)

ranking:
    rank features by |g_j|

Note on penalty policy:
    The transformer experiments used uniform_penalty=True, unlike the adaptive
    penalty used by gated_mlp. This is a protocol mismatch.

Follow-up verification (2026-04-03):
    Re-ran transformer with adaptive penalty to check if this affected the conclusion.
    Results: both uniform and adaptive penalty yield best-k ≈ 0, AUC ≈ 0.5 (random).
    The penalty policy was NOT the cause of failure. The transformer architecture
    itself is unsuitable for this synthetic feature-selection benchmark.

    | Task | Uniform best-k | Adaptive best-k |
    |------|----------------|-----------------|
    | xor_m128 | 0.0000 | 0.0000 |
    | ring_m128 | 0.1667 | 0.0833 |
    | ring+xor_m256 | 0.0000 | 0.0000 |

    See: `transformer_penalty_ablation.json`
```

Important clarification:

- this is transformer-based
- but selection is still done by the global scalar gate, not by transformer attention itself

#### C. Masked-Pretrained Transformer

Pseudocode:

```text
pretraining:
    randomly mask input features
    disable gate effect during reconstruction pretraining
    reconstruct masked raw features from encoded feature tokens

fine-tuning:
    switch to standard gated ADMM training
```

### 2.3 Result

Source:

- [`backbone_tier2_synthetic_full_20260327.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\backbone_tier2_synthetic_full_20260327.json)

Protocol:

- full synthetic grid
- datasets:
  - `xor`
  - `ring`
  - `ring+xor`
  - `ring+xor+sum`
- `n=1000`
- `6-fold CV`
- metrics:
  - `best-k`
  - `best-2k`
  - `AUC`
  - `AUPRC`

Overall means:

| Method | Mean best-k | Mean best-2k | Mean AUC | Mean AUPRC |
|---|---:|---:|---:|---:|
| `gated_mlp` | `0.6278` | `0.6656` | `0.6605` | `0.6822` |
| `gated_token_transformer` | `0.2479` | `0.3146` | `0.5544` | `0.5693` |
| `gated_token_transformer_pretrained` | `0.2094` | `0.2792` | `0.5481` | `0.5626` |

Selected tasks:

| Task | `gated_mlp` best-k / AUC | `transformer` best-k / AUC | `pretrained transformer` best-k / AUC |
|---|---:|---:|---:|
| `xor_m128` | `1.0000 / 0.8742` | `0.0000 / 0.5000` | `0.0000 / 0.4945` |
| `ring_m128` | `0.5000 / 0.5129` | `0.0000 / 0.5000` | `0.0833 / 0.5000` |
| `ring+xor_m256` | `0.6250 / 0.5351` | `0.0417 / 0.5000` | `0.0000 / 0.5000` |
| `ring+xor+sum_m256` | `0.6667 / 0.5853` | `0.0556 / 0.5000` | `0.0278 / 0.5000` |

### 2.4 Conclusion

- The transformer backbone is a negative result.
- Pretraining does not rescue it.
- There is no realistic basis for promoting this branch into the main method.

Current decision:

- keep `gated_mlp` as the default backbone
- keep transformer variants only as exploratory negative results

## 3. Tier-2 Baselines

### 3.1 Idea

Question:

> If we compare against encoder-style or embedded neural baselines, does the plain gated MLP still hold up?

Tier-2 set:

- `FSNet`
- `E2E-FS`
- `CAE`
- `TabNet`

### 3.2 Implementation

#### A. FSNet

Implementation path:

- [`custom_admm/src/core.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\core.py)
- [`custom_admm/src/fsnet.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\fsnet.py)

Pseudocode:

```text
Architecture:
    n_select = min(2k, m)
    selector = concrete selector over m input features
    predictor = downstream classifier on the selected representation
    auxiliary branch = encoder + decoder + reconstruction head back to the original m features
    initialization = histogram summary U for each feature initializes selector/reconstruction weights

Fit:
    for epoch = 1..T:
        anneal selector temperature
        sample soft selector matrix M in R^(m x n_select)
        X_subset = X @ M
        y_hat = predictor(X_subset)
        X_recon = reconstruction_branch(X_subset)
        loss = classification_loss(y_hat, y) + lambda * mse(X_recon, X)
        update selector, predictor, and reconstruction parameters jointly

Inference:
    convert selector to hard picks
    predict from the selected subset

Scoring:
    feature_score[j] = mean selector probability assigned to feature j across selector columns

Note on evaluation protocol:
    The actual implementation trains separate models for k and 2k features
    (fsnet_k and fsnet_2k). Best-k recovery uses scores from the k model,
    best-2k recovery uses scores from the 2k model. Predictions come from
    the 2k model when available. This differs from single-model methods.
```

#### B. E2E-FS

Implementation path:

- [`custom_admm/src/e2efs_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\e2efs_wrapper.py)
- [`vendor_pkgs/e2efs/e2efs_modules.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\vendor_pkgs\e2efs\e2efs_modules.py)
- [`vendor_pkgs/e2efs/networks.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\vendor_pkgs\e2efs\networks.py)

Pseudocode:

```text
Architecture:
    selector = elementwise mask s in [0, 1]^m applied to the original input
    predictor = vendor three_layer_nn:
        Linear(m, 50) -> BatchNorm -> SiLU
        -> Linear(50, 25) -> BatchNorm -> SiLU
        -> Linear(25, 10) -> BatchNorm -> SiLU
        -> Linear(10, n_classes)
    optimization split = one optimizer for the predictor, one optimizer for the mask

Fit:
    split X, y into fit and validation folds
    fit selector_k end-to-end for target size k
    for each step:
        masked_x = x * s
        task_loss = classifier_loss(predictor(masked_x), y)
        mask_penalty = sparsity/control penalty on the mask
        update predictor with task gradient
        update mask with combined task-gradient + penalty-gradient
        clamp mask entries into [0, 1]
        once the number of alive features is <= k, freeze the mask and finish fitting the predictor
    harden the final mask to a top-k support

Inference:
    y_train_hat = selector_k.predict(X_train)
    y_hat = selector_k.predict(X_test)

Scoring:
    scores_k = selector_k.get_mask()
    if 2k differs from k:
        fit a second selector_2k on the same split
        scores_2k = selector_2k.get_mask()
    else:
        scores_2k = scores_k
    use selector_k / selector_2k masks as ranking scores for best-k / best-2k recovery
```

#### C. CAE

Implementation path:

- [`custom_admm/src/cae_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\cae_wrapper.py)
- [`vendor_pkgs/concrete_autoencoder/__init__.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\vendor_pkgs\concrete_autoencoder\__init__.py)

Pseudocode:

```text
Architecture:
    selector = k concrete selector rows over m input features
    predictor head =
        Linear(k, 32) -> ReLU -> Dropout
        -> Linear(32, 32) -> ReLU -> Dropout
        -> Linear(32, n_classes or 1)

Fit:
    train one selector for k features
    optionally train a second selector for 2k features
    for epoch = 1..T:
        sample Gumbel-softmax selector rows
        selected_representation = x @ selector_matrix^T
        logits = predictor(selected_representation)
        optimize prediction loss
        decay selector temperature

Inference:
    hard-pick one feature per selector row
    predict with the same MLP head
    use the 2k selector for reported probabilities if it is trained; otherwise use the k selector

Scoring:
    feature_score[j] = sum over selector rows of softmax(logits_row)[j]
    return soft scores for best-k and, if separately trained, best-2k ranking
```

CAE local vs CAE original:

| Aspect | CAE local (`cae_wrapper.py`) | CAE original (`core.py`) |
|---|---|---|
| Framework | local PyTorch reimplementation | Keras + `concrete_autoencoder` package |
| Training runs | train one selector for `k`, optionally a second selector for `2k` | same high-level pattern: one run for `k`, another for `2k` |
| Prediction head | fixed PyTorch MLP: `Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear` | Keras head with `GaussianNoise`, `Dense`, `Dropout`, `LeakyReLU`, final sigmoid |
| Optimization | manual Adam loop, early stopping, temperature decay `10 * 0.95^epoch` floored at `0.1` | package-controlled training with `start_temp=10`, `min_temp=0.01`, `num_epochs=300/30`, `lr=1e-4` |
| Feature scores returned to benchmark | soft scores: `sum_row softmax(logits)[row, j]` | hard support only: selected indices are set to `1`, all others `0` |
| Prediction source | predictions come from the `2k` selector if it is trained, otherwise from the `k` selector | predictions also come from the `2k` selector after the second fit |

Important caveat:

- the benchmark uses the local CAE-style implementation above, not the exact old Keras package
- reason: the historical Keras CAE package depends on legacy backend calls such as `K.set_learning_phase`, `K.update`, `K.in_train_phase`, and `K.random_uniform`, which are absent in the current Keras backend in this environment
- so the local PyTorch version is the runnable, reproducible CAE-style proxy used by the Tier-2 benchmark

#### D. TabNet

Implementation path:

- [`custom_admm/src/tabnet_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\tabnet_wrapper.py)
- [`vendor_pkgs/pytorch_tabnet/tab_network.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\vendor_pkgs\pytorch_tabnet\tab_network.py)
- [`vendor_pkgs/pytorch_tabnet/abstract_model.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\vendor_pkgs\pytorch_tabnet\abstract_model.py)

Pseudocode:

```text
Architecture:
    input preprocessing = batch normalization
    encoder = 5 decision steps
    each step contains:
        attentive transformer = Linear(n_a -> m) + ghost batch norm + sparsemax mask
        feature transformer = shared GLU block + step-specific GLU block
    widths = n_d = 16 decision channels, n_a = 16 attention channels
    classifier head = linear map from summed decision states to class logits

Fit:
    split X, y into fit and validation folds
    for each step t = 1..5:
        produce sparse mask_t from the current attention state and running prior
        apply masked_x_t = mask_t * x
        transform masked_x_t with the step feature transformer
        keep the first n_d channels as decision output
        pass the last n_a channels to the next attention step
        update the prior so later steps reuse features less
    sum decision outputs across steps
    map the sum to class logits
    optimize cross-entropy minus sparse-mask regularization
    select checkpoint by validation AUC

Inference:
    y_train_hat = predict_proba(X_train)
    y_hat = predict_proba(X_test)

Scoring:
    scores = global feature_importances_ from summed explain masks over samples
    scores_2k = scores
    rank features once by the global importance vector
    take top-k and top-2k prefixes from the same ranking
```

Important clarification:

- TabNet is not transformer-based
- it is an attentive tabular encoder with sequential feature masks

### 3.3 Result

Source:

- [`backbone_tier2_synthetic_full_20260327.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\backbone_tier2_synthetic_full_20260327.json)

Overall means:

| Method | Mean best-k | Mean best-2k | Mean AUC | Mean AUPRC |
|---|---:|---:|---:|---:|
| `gated_mlp` | `0.6278` | `0.6656` | `0.6605` | `0.6822` |
| `tabnet` | `0.2851` | `0.3476` | `0.5995` | `0.6191` |
| `gated_token_transformer` | `0.2479` | `0.3146` | `0.5544` | `0.5693` |
| `gated_token_transformer_pretrained` | `0.2094` | `0.2792` | `0.5481` | `0.5626` |
| `cae` | `0.1861` | `0.2667` | `0.5236` | `0.5496` |
| `e2efs` | `0.1767` | `0.2559` | `0.5117` | `0.5359` |
| `fsnet` | `0.1667` | `0.2590` | `0.5621` | `0.5867` |

Selected tasks:

| Task | `gated_mlp` | `tabnet` | `fsnet` | `e2efs` | `cae` |
|---|---:|---:|---:|---:|---:|
| `xor_m128` best-k / AUC | `1.0000 / 0.8742` | `0.1667 / 0.5556` | `0.0000 / 0.4902` | `0.0833 / 0.5200` | `0.0000 / 0.5251` |
| `ring_m128` best-k / AUC | `0.5000 / 0.5129` | `0.0833 / 0.4992` | `0.0000 / 0.5075` | `0.0833 / 0.4745` | `0.0000 / 0.5088` |
| `ring+xor_m256` best-k / AUC | `0.6250 / 0.5351` | `0.0417 / 0.5154` | `0.0000 / 0.4926` | `0.0000 / 0.4965` | `0.0417 / 0.4855` |
| `ring+xor+sum_m256` best-k / AUC | `0.6667 / 0.5853` | `0.0556 / 0.5442` | `0.0000 / 0.4719` | `0.0000 / 0.5026` | `0.0833 / 0.5728` |

### 3.4 Conclusion

- `gated_mlp` stays clearly ahead of all Tier-2 baselines.
- `TabNet` is the strongest Tier-2 baseline and should be the one to keep in discussion.
- `FSNet`, `E2E-FS`, and `CAE` do not threaten the main method on this grid.

Current decision:

- keep `TabNet` as the most relevant encoder-style baseline
- treat the others as lower-priority support baselines

## 4. Training Method

### 4.1 Idea

Question:

> Would neural feature selection work better if we first expand each feature into a richer local representation, then perform feature selection, then classify?

Hypothesis:

```text
feature expansion
-> feature selection
-> MLP
```

might outperform:

```text
feature selection
-> MLP
```

### 4.2 Implementation

#### A. Baseline: Select Then MLP

Implementation:

- `select_then_mlp`

Pseudocode:

```text
input x
apply gate g on raw features
x_gated = x * g
y_hat = MLP(x_gated)
```

#### B. Expansion Then Selection Then MLP

Implementation:

- `ExpandedFeatureSelectionMLP` in [`mentor_models.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\mentor_models.py)

Pseudocode:

```text
parameters:
    expansion matrix W_exp[j] for each original feature j
    global gate g in R^m
    MLP on flattened expanded features

forward:
    for each original feature j:
        token_j = x[j] * W_exp[j]
        token_j = token_j * g[j]
        token_j = activation(token_j)  # CRITICAL: nonlinearity prevents algebraic collapse
    z = concatenate(token_1, ..., token_m)
    y_hat = MLP(z)

ADMM score:
    score_j = ||W_exp[j]||_2

ranking:
    rank by |g_j|
```

Important clarification:

- this is still original-feature-level selection
- one gate `g[j]` controls the whole expanded block for feature `j`

Tested widths:

- `expand4`
- `expand8`
- `expand16`

### 4.3 Result

Source:

- [`training_order_synthetic_full_20260329.json`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\training_order_synthetic_full_20260329.json)

Overall means:

| Method | Mean best-k | Mean best-2k | Mean AUC | Mean AUPRC |
|---|---:|---:|---:|---:|
| `select_then_mlp` | `0.6090` | `0.6483` | `0.6628` | `0.6844` |
| `expand4_then_select_then_mlp` | `0.3694` | `0.4181` | `0.6221` | `0.6486` |
| `expand8_then_select_then_mlp` | `0.4184` | `0.4573` | `0.6383` | `0.6621` |
| `expand16_then_select_then_mlp` | `0.4181` | `0.4597` | `0.6537` | `0.6774` |

Selected tasks:

| Task | `select_then_mlp` | `expand4` | `expand8` | `expand16` |
|---|---:|---:|---:|---:|
| `xor_m128` best-k / AUC | `1.0000 / 0.8521` | `0.5833 / 0.8310` | `0.5000 / 0.7606` | `0.6667 / 0.8249` |
| `ring_m128` best-k / AUC | `0.6667 / 0.4989` | `0.0000 / 0.5277` | `0.0833 / 0.5305` | `0.0833 / 0.5222` |
| `ring+xor_m256` best-k / AUC | `0.4167 / 0.5222` | `0.0417 / 0.4798` | `0.0417 / 0.4921` | `0.1667 / 0.5565` |
| `ring+xor+sum_m256` best-k / AUC | `0.5833 / 0.5697` | `0.3333 / 0.6313` | `0.3333 / 0.6261` | `0.3889 / 0.6367` |

Win counts:

| Method | Best-k wins | AUC wins |
|---|---:|---:|
| `select_then_mlp` | `35` | `16` |
| `expand4_then_select_then_mlp` | `13` | `4` |
| `expand8_then_select_then_mlp` | `15` | `8` |
| `expand16_then_select_then_mlp` | `12` | `12` |

### 4.4 Conclusion

- The evidence does not support replacing the current pipeline with expansion-before-selection.
- `select_then_mlp` remains clearly best on the main metric `best-k`.
- `expand16` is the strongest expansion variant, but still not good enough to replace the baseline.
- Expansion can occasionally help `AUC` on mixed tasks, but that is not enough for this project.

Current decision:

- keep `select_then_mlp` as the main pipeline
- keep expansion-before-selection only as an internal ablation

## Protocol Notes and Caveats

### Training Budget Disparities (Tier-2 Baselines)

| Method | Training Epochs | Notes |
|--------|-----------------|-------|
| ADMM methods (gated_mlp, transformer) | 240 | 60 warmup, 180 ADMM |
| TabNet | 100 | Early stopping with patience 20 |
| E2E-FS | 200 | Per selector (k and 2k separately) |
| CAE | 300 | Per selector (k and 2k separately) |
| FSNet | 2000 | Much longer training |

**Impact**: These disparities actually *favor* Tier-2 baselines, yet they still underperform. The conclusion that gated_mlp outperforms Tier-2 baselines is strengthened, not weakened, by this observation.

### Evaluation Protocol Inconsistencies

Some Tier-2 baselines train separate models for k and 2k features:
- **FSNet**: Separate fsnet_k and fsnet_2k models
- **E2E-FS**: Separate selector_k and selector_2k
- **CAE**: Separate selector for k and 2k

These methods report:
- best-k from the k-trained model
- AUC/AUPRC from the 2k-trained model (when available)

**Impact**: This gives Tier-2 baselines an advantage (2k features for prediction), yet they still underperform gated_mlp which uses a single model for all metrics.

## Final Recommendation

### Keep as Main Method

- `gated_mlp`
- unbounded global scalar gate
- ADMM + Ratio Norm
- `select_then_mlp`

### Keep as Main Comparator

- `TabNet`

### Keep as Internal Ablations

- `sigmoid/bounded gate`
- `gated_token_transformer`
- `gated_token_transformer_pretrained`
- `expand4/8/16_then_select_then_mlp`

### Do Not Promote Into Mainline

- transformer backbone branch
- pretraining branch
- expansion-before-selection branch

## Short Verbal Summary

> We tested the mentor directions from three angles. On gating, after fixing the bounded-gate ADMM mismatch, the unbounded linear gate still performs slightly better and remains the default. On backbone, the plain gated MLP remains clearly stronger than both the transformer variants and all Tier-2 encoder-style baselines; among those, TabNet is the strongest but still substantially worse than the main method. On training order, expansion-before-selection does not beat the simpler select-then-MLP pipeline on best-k, even though larger expansions can occasionally improve AUC on mixed tasks. Overall, the simplest global-gate MLP with ADMM remains the strongest and cleanest implementation.
