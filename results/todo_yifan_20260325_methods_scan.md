# TODO_Yifan Follow-up (2026-03-25)

## 1. Are the currently reported benchmark results using the same model structure?

No, not in the strict "same backbone / same optimizer / same tuning budget" sense.

- Main cross-method benchmark: each baseline uses its standard implementation/configuration.
- SADMM-FS architecture sweeps: reported separately as internal ablations only.
- Current paper text already states this explicitly in `deep-reg-paper/uai2026/main.tex`.

This means the paper should **not** claim a controlled same-architecture comparison across all methods yet.

## 2. Additional MLP-based embedded feature selection methods

### E2E-FS

- Type: global embedded feature selection for neural networks.
- Why it matters: directly learns a mask/ranking and is closer to the "select a subset of features" setup than many explanation-only methods.
- Code status: good.
- Practicality: high.
- Notes:
  - Public Python package `e2efs` exists.
  - Public GitHub repo is linked from PyPI.
  - Exposes `get_mask()` and `get_ranking()`.
  - Better immediate candidate than older explanation-oriented selectors.

### INVASE

- Type: actor-critic neural selector.
- Why it matters: classic neural embedded selector with official code.
- Code status: good.
- Practicality: medium.
- Notes:
  - Official GitHub repo exists.
  - PyPI package also exists via `vanderschaarlab/INVASE`.
  - Original method is instance-wise selection, so global ranking would need aggregation across samples.
  - This makes it less clean than E2E-FS for a global top-k benchmark.

### L2X

- Type: information-theoretic selector/explainer with a learned selector network.
- Why it matters: historically important neural selector.
- Code status: usable, but old.
- Practicality: low to medium.
- Notes:
  - Official GitHub repo exists.
  - Implementation is TensorFlow/Keras-era code.
  - Primarily designed for instance-wise explanations rather than a clean global ranking benchmark.

### Learnable Drop Layer

- Type: neural embedded selector via a learnable drop layer.
- Why it matters: conceptually relevant to gating.
- Code status: weak.
- Practicality: low.
- Notes:
  - I found the paper, but not a reliable public implementation in quick GitHub-targeted search.
  - Not a good immediate baseline unless you want to re-implement it yourself.

### Recommendation for TODO 2

If you want one more runnable MLP-based embedded baseline with reasonable implementation cost:

1. Add `E2E-FS` first.
2. Add `INVASE` only if you are willing to treat it as an instance-wise selector and define a global aggregation rule.
3. Do not prioritize `L2X` or Learnable Drop Layer for the paper mainline.

## 3. Transformer / embedding-space feature selection baselines with runnable code

### FT-Transformer attention baseline

- Best immediate option.
- Why it matters: it is actually a transformer-based tabular feature-selection pipeline, not just a transformer backbone.
- Code status: strong.
- Practicality: high.
- Notes:
  - `vcherepanova/tabular-feature-selection` provides a full benchmark repo.
  - It supports `train_deep_model.py`.
  - It explicitly documents `model=ft_transformer_attention` for attention-map-based feature importance.
  - It also supports `hyp.regularization=deep_lasso` and `hyp.regularization=first_lasso` with `model=ft_transformer`.

### FT-Transformer official backbone

- Type: official tabular transformer implementation.
- Code status: strong.
- Practicality: medium.
- Notes:
  - `yandex-research/rtdl-revisiting-models` is the official FT-Transformer repo.
  - It is the right backbone source if you want a controlled custom implementation.
  - It is not by itself a feature-selection repo; you still need an attribution or sparsity mechanism on top.

### TabNet

- Type: sparse attention model for tabular data (not a vanilla transformer, but close in spirit for embedding/attention-space selection).
- Code status: very strong.
- Practicality: high.
- Notes:
  - Official `dreamquark-ai/tabnet` repo exists and is maintained.
  - It exposes sparse attention and feature importance computation.
  - It supports `grouped_features`, which is useful if you later move from vector gates to grouped gates.
- Caveat:
  - It is attention-based tabular selection, but not a transformer in the FT-Transformer sense.

### Recommendation for TODO 3

If the question is "is there a transformer-like baseline with code that we can really run?", the answer is yes:

1. Use `vcherepanova/tabular-feature-selection` with `model=ft_transformer_attention` as the cleanest transformer baseline.
2. If you want a second attention-based baseline, use `TabNet`.
3. Do not spend time on custom TabTransformer feature-importance code unless you specifically want a new implementation project.

## 4. Gating design options

The key constraint is ADMM compatibility. Anything that removes the single global gate vector makes the comparison harder and the optimization less clean.

### A. Global vector gate (current safe path)

- Form: `x' = x * g`, `g in R^m`.
- Pros:
  - Cleanest feature semantics.
  - Directly compatible with current ADMM solver.
  - Best choice for main paper comparisons.
- Cons:
  - Cannot model interactions directly.

### B. Group gate / block gate

- Form: partition features into groups, learn one scalar gate per group, then broadcast within group.
- Pros:
  - Still ADMM-compatible.
  - More robust when features are correlated or engineered in bundles.
  - Natural next step if you want more structure without losing interpretability.
- Cons:
  - Requires a grouping rule.

### C. Diagonal + low-rank gate

- Form: `x' = x * g + x W`, with low-rank `W = U V^T`, or equivalently a diagonal gate plus a small interaction term.
- Pros:
  - Keeps a global feature gate for selection.
  - Adds limited cross-feature interaction capacity.
  - You can still run ADMM on `g` only and train `U,V` normally.
- Cons:
  - Attribution becomes less pure once `W` is active.
  - Needs careful ablation to show the gain really comes from interactions.

### D. Full matrix gate

- Form: `x' = G x`.
- Pros:
  - Most expressive.
- Cons:
  - Feature selection semantics become ambiguous because features are mixed.
  - ADMM no longer acts on a single interpretable per-feature variable.
  - Not a good next experiment for the paper.

### E. Sample-conditioned gate

- Form: `g(x)` from a gate network.
- Pros:
  - Very expressive.
- Cons:
  - No single global gate vector.
  - Hard to connect fairly to the current ADMM formulation.
  - More like instance-wise explanation than global feature selection.

### F. Hard top-k / hard-concrete gate

- Form: stochastic binary or top-k gates.
- Pros:
  - Direct control over sparsity/cardinality.
- Cons:
  - Different optimization regime.
  - Adds another fairness problem if SADMM-FS is compared against soft gate baselines.

### Recommendation for TODO 4

If the goal is to extend gating without breaking the current optimization story:

1. Try **group gate** first.
2. Then try **diagonal + low-rank gate**.
3. Do **not** prioritize full matrix gates or sample-conditioned gates for the main paper.

## Bottom line

- Current reported cross-method benchmark is **not** a same-model-structure comparison.
- Best new MLP embedded baseline: **E2E-FS**.
- Best transformer-style runnable baseline: **FT-Transformer attention** from `vcherepanova/tabular-feature-selection`.
- Best next gating extension: **group gate**, then **diagonal + low-rank gate**.

## Sources

- E2E-FS paper: https://pubmed.ncbi.nlm.nih.gov/37015369/
- E2E-FS package / repo link: https://pypi.org/project/e2efs/
- INVASE repo: https://github.com/jsyoon0823/INVASE
- INVASE package: https://github.com/vanderschaarlab/INVASE
- L2X repo: https://github.com/Jianbo-Lab/L2X
- Tabular feature selection benchmark: https://github.com/vcherepanova/tabular-feature-selection
- FT-Transformer official repo: https://github.com/yandex-research/rtdl-revisiting-models
- TabNet official repo: https://github.com/dreamquark-ai/tabnet
- Learnable drop layer paper: https://academic.oup.com/jigpal/article/doi/10.1093/jigpal/jzae062/7689640
