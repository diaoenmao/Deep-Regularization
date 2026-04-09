# Gating Fix and Rerun (2026-03-30)

## What Was Fixed

The previous `sigmoid_gate_mlp` experiment had a real implementation mismatch:

- the forward pass used `sigmoid(raw_gate)`
- but ADMM sparsified `raw_gate` directly

That meant:

```text
raw_gate -> 0
does not imply
effective_gate -> 0
```

because `sigmoid(0) = 0.5`.

The fix in [`admm_input_group_wrapper.py`](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py) was:

- add gate-parameter / effective-gate conversion:
  - [admm_input_group_wrapper.py:226](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:226)
  - [admm_input_group_wrapper.py:231](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:231)
  - [admm_input_group_wrapper.py:237](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:237)
- move ADMM consensus to effective-gate space:
  - [admm_input_group_wrapper.py:568](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:568)
  - [admm_input_group_wrapper.py:584](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:584)
  - [admm_input_group_wrapper.py:666](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:666)
  - [admm_input_group_wrapper.py:772](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:772)
  - [admm_input_group_wrapper.py:785](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:785)
  - [admm_input_group_wrapper.py:835](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\src\admm_input_group_wrapper.py:835)

Practical effect:

- ADMM now acts on the same gate quantity used in the forward pass
- bounded-gate results are now interpretable as a real bounded-gate ablation

## Rerun Outputs

Quick rerun:

- [mentor_gating_quick_20260330_fixed.json](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\mentor_gating_quick_20260330_fixed.json)

Fuller rerun across the gating task set:

- [mentor_gating_full_20260330_fixed.json](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\mentor_gating_full_20260330_fixed.json)

## Full Gating Rerun Results

Tasks:

- `xor_m128`
- `ring_m128`
- `ring+xor_m256`

Protocol:

- `6` folds
- `240` epochs
- `60` warmup epochs

### Per Task

| Task | Linear gate best-k / AUC | Sigmoid gate best-k / AUC | Readout |
|---|---:|---:|---|
| `xor_m128` | `1.0000 / 0.8532` | `1.0000 / 0.8448` | Essentially tied on best-k; linear slightly better on AUC |
| `ring_m128` | `0.6667 / 0.5126` | `0.5833 / 0.4905` | Linear better |
| `ring+xor_m256` | `0.5417 / 0.5592` | `0.5417 / 0.5408` | Tie on best-k; linear better on AUC |

### Overall Mean Across Tasks

| Method | Mean best-k | Mean AUC |
|---|---:|---:|
| `linear_gate_mlp` | `0.7361` | `0.6417` |
| `sigmoid_gate_mlp` | `0.7083` | `0.6254` |

## Conclusion

After fixing the implementation mismatch:

- `sigmoid_gate_mlp` is still not better than the unbounded linear gate
- the gap is smaller than the earlier critique suggested
- but the current evidence still favors the unbounded gate as the default

So the current report position should be:

- keep the unbounded global gate as the main method
- keep sigmoid gating as a bounded-gate ablation
- describe the old mismatch as fixed, and cite the rerun rather than the old quick pilot
