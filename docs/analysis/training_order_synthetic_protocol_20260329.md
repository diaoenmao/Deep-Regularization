# Training-Order Synthetic Protocol

## Goal

Test the mentor-requested hypothesis that:

- `feature expansion -> feature selection -> MLP`

may work better than the current order:

- `feature selection -> MLP`

## Methods

- `select_then_mlp`
  - baseline gated MLP
- `expand4_then_select_then_mlp`
  - expand each feature to 4 channels before gating
- `expand8_then_select_then_mlp`
  - expand each feature to 8 channels before gating
- `expand16_then_select_then_mlp`
  - expand each feature to 16 channels before gating

## Shared Protocol

- datasets:
  - `xor`
  - `ring`
  - `ring+xor`
  - `ring+xor+sum`
- sample size:
  - `n = 1000`
- folds:
  - `6-fold CV`
- feature permutation:
  - enabled per fold
- metrics:
  - `best-k`
  - `best-2k`
  - `AUC`
  - `AUPRC`
- ADMM:
  - enabled
- penalty:
  - Ratio Norm
- optimizer:
  - Adam
- feature dropout:
  - `0.6`

## Training Budget

- planned full run:
  - `epochs = 240`
  - `warmup_epochs = 60`

## Output

- result JSON:
  - `custom_admm/results/mentor_axes/training_order_synthetic_full_*.json`
- task queue logs:
  - `custom_admm/results/mentor_axes/training_order_task_logs/`

## Interpretation

This is an internal method-design experiment. The main question is not whether
expansion variants beat external baselines, but whether expansion before
selection improves feature recovery relative to the current SADMM-FS order.
