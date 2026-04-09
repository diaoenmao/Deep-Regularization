# Training-Order Full Synthetic Summary

Source:

- [training_order_synthetic_full_20260329.json](C:\Users\12425\Documents\Projects\NEW_Pruning_20251110\custom_admm\results\mentor_axes\training_order_synthetic_full_20260329.json)

## Compared Methods

- `select_then_mlp`
- `expand4_then_select_then_mlp`
- `expand8_then_select_then_mlp`
- `expand16_then_select_then_mlp`

All methods were evaluated on the full synthetic grid:

- `xor`
- `ring`
- `ring+xor`
- `ring+xor+sum`

with:

- `n = 1000`
- `6-fold CV`
- feature permutation
- metrics: `best-k`, `best-2k`, `AUC`, `AUPRC`

## Overall Averages

| Method | Mean best-k | Mean best-2k | Mean AUC | Mean AUPRC |
|---|---:|---:|---:|---:|
| `select_then_mlp` | `0.6090` | `0.6483` | `0.6628` | `0.6844` |
| `expand4_then_select_then_mlp` | `0.3694` | `0.4181` | `0.6221` | `0.6486` |
| `expand8_then_select_then_mlp` | `0.4184` | `0.4573` | `0.6383` | `0.6621` |
| `expand16_then_select_then_mlp` | `0.4181` | `0.4597` | `0.6537` | `0.6774` |

## By Dataset Family

### XOR

| Method | Mean best-k | Mean AUC |
|---|---:|---:|
| `select_then_mlp` | `0.7576` | `0.8098` |
| `expand4_then_select_then_mlp` | `0.4773` | `0.7310` |
| `expand8_then_select_then_mlp` | `0.5379` | `0.7578` |
| `expand16_then_select_then_mlp` | `0.5909` | `0.7929` |

### Ring

| Method | Mean best-k | Mean AUC |
|---|---:|---:|
| `select_then_mlp` | `0.5000` | `0.5355` |
| `expand4_then_select_then_mlp` | `0.2593` | `0.5286` |
| `expand8_then_select_then_mlp` | `0.3426` | `0.5372` |
| `expand16_then_select_then_mlp` | `0.2685` | `0.5324` |

### Ring+XOR

| Method | Mean best-k | Mean AUC |
|---|---:|---:|
| `select_then_mlp` | `0.5000` | `0.6284` |
| `expand4_then_select_then_mlp` | `0.2250` | `0.5553` |
| `expand8_then_select_then_mlp` | `0.2625` | `0.5859` |
| `expand16_then_select_then_mlp` | `0.2833` | `0.6016` |

### Ring+XOR+Sum

| Method | Mean best-k | Mean AUC |
|---|---:|---:|
| `select_then_mlp` | `0.6528` | `0.6500` |
| `expand4_then_select_then_mlp` | `0.4944` | `0.6533` |
| `expand8_then_select_then_mlp` | `0.5111` | `0.6504` |
| `expand16_then_select_then_mlp` | `0.4972` | `0.6619` |

## Win Counts

Counted per synthetic task over the full grid.

| Method | Best-k wins | AUC wins |
|---|---:|---:|
| `select_then_mlp` | `35` | `16` |
| `expand4_then_select_then_mlp` | `13` | `4` |
| `expand8_then_select_then_mlp` | `15` | `8` |
| `expand16_then_select_then_mlp` | `12` | `12` |

## Selected Tasks

| Task | Method | best-k | AUC |
|---|---|---:|---:|
| `xor_m128` | `select_then_mlp` | `1.0000` | `0.8521` |
| `xor_m128` | `expand4_then_select_then_mlp` | `0.5833` | `0.8310` |
| `xor_m128` | `expand8_then_select_then_mlp` | `0.5000` | `0.7606` |
| `xor_m128` | `expand16_then_select_then_mlp` | `0.6667` | `0.8249` |
| `xor_m512` | `select_then_mlp` | `0.5000` | `0.4835` |
| `xor_m512` | `expand4_then_select_then_mlp` | `0.0000` | `0.4769` |
| `xor_m512` | `expand8_then_select_then_mlp` | `0.0000` | `0.4778` |
| `xor_m512` | `expand16_then_select_then_mlp` | `0.1667` | `0.5436` |
| `ring_m32` | `select_then_mlp` | `1.0000` | `0.5672` |
| `ring_m32` | `expand4_then_select_then_mlp` | `0.5833` | `0.5448` |
| `ring_m32` | `expand8_then_select_then_mlp` | `0.8333` | `0.5550` |
| `ring_m32` | `expand16_then_select_then_mlp` | `0.5833` | `0.5400` |
| `ring_m128` | `select_then_mlp` | `0.6667` | `0.4989` |
| `ring_m128` | `expand4_then_select_then_mlp` | `0.0000` | `0.5277` |
| `ring_m128` | `expand8_then_select_then_mlp` | `0.0833` | `0.5305` |
| `ring_m128` | `expand16_then_select_then_mlp` | `0.0833` | `0.5222` |
| `ring+xor_m256` | `select_then_mlp` | `0.4167` | `0.5222` |
| `ring+xor_m256` | `expand4_then_select_then_mlp` | `0.0417` | `0.4798` |
| `ring+xor_m256` | `expand8_then_select_then_mlp` | `0.0417` | `0.4921` |
| `ring+xor_m256` | `expand16_then_select_then_mlp` | `0.1667` | `0.5565` |
| `ring+xor+sum_m256` | `select_then_mlp` | `0.5833` | `0.5697` |
| `ring+xor+sum_m256` | `expand4_then_select_then_mlp` | `0.3333` | `0.6313` |
| `ring+xor+sum_m256` | `expand8_then_select_then_mlp` | `0.3333` | `0.6261` |
| `ring+xor+sum_m256` | `expand16_then_select_then_mlp` | `0.3889` | `0.6367` |

## Main Takeaway

The current evidence does **not** support replacing the current order with
`feature expansion -> feature selection -> MLP`.

The most defensible conclusion is:

- `select_then_mlp` remains the best default method for feature recovery
- expansion variants can sometimes improve `AUC` on harder mixed tasks
- but they do not improve `best-k` consistently, which is the primary metric for this work

Among expansion variants:

- `expand16_then_select_then_mlp` is the strongest of the three
- but it still remains clearly below `select_then_mlp` on overall `best-k`

## Recommendation

- Keep `select_then_mlp` as the main method
- If this line is reported, present it as an internal ablation:
  - expansion before selection can slightly help predictive metrics in some tasks
  - but it weakens feature recovery and should not replace the main pipeline
