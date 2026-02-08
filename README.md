# SADMM: Scope-Driven Stochastic ADMM for Neural Network Pruning

A novel neural network pruning framework that integrates **Ratio Norm ($L_1/L_2$)** regularization with **Taylor expansion-based importance scores** (including Wanda) into an ADMM optimization loop. The framework supports three pruning scopes (global, layer, neuron) across three optimizer families (ADMM, Lasso, Ppercent), totaling **9 optimizer variants × 4 score types = 36 method combinations**.

---

## Key Results (MNIST, CNN3 ~7.5M params)

| Goal | Method | Score | Config | Accuracy | Sparsity | Compression |
|------|--------|-------|--------|----------|----------|-------------|
| **Highest Accuracy** | Ppercent_global | Second-Order | p=10 | **97.67%** | 35.1% | 1.5× |
| **Best Compression** | ADMM_neuron | Magnitude | C=0.03 | **94.66%** | **99.0%** | **100×** |
| **Best Tradeoff** | ADMM_global | Magnitude | C=0.03 | 92.80% | 98.7% | 77× |
| **Most Robust** | Lasso_neuron | Magnitude | any C | 96.95% | 77.3% | 4.4× |

> 360 configurations tested across 9 optimizers × 4 scores × 10 hyperparameter values.
> Total training time: ~30.5 hours. Full report: [results/EXPERIMENT_REPORT.md](results/EXPERIMENT_REPORT.md).

---

## Mathematical Framework

The optimization objective is:

$$\min_q \; f(q) + C \cdot \frac{\|s \odot q\|_1}{\|s \odot q\|_2}$$

where $s$ is the importance score (Wanda/Taylor) and $C$ controls sparsity. The ADMM splitting introduces auxiliary variables $y, z$ with dual variables $v, w$:

| Variable | Update Rule | Purpose |
|----------|-------------|---------|
| $q_k$ (weights) | Gradient + dual averaging | Minimizes loss |
| $y_k$ (auxiliary) | Cubic solver ($\tau^3 - \tau - D_k = 0$) | Enforces Ratio Norm |
| $z_k$ (sparse) | Soft-thresholding on $s \odot q_k$ | Promotes sparsity |
| $v_k, w_k$ (dual) | Gradient ascent on constraints | Ensures $q = y = z$ |

The cubic equation uses Cardano's formula with trigonometric fallback for numerical stability.

---

## Project Structure

```
├── network/
│   └── cnn3.py                 # 2-conv + 2-FC CNN for MNIST (~7.5M params)
├── optimizer/                  # 9 pruning optimizers
│   ├── ADMM_{global,layer,neuron}.py   # ADMM with Ratio Norm (L1/L2)
│   ├── lasso_{global,layer,neuron}.py  # L1-regularized pruning
│   ├── ppercent_{global,layer,neuron}.py # Percentile-based hard pruning
│   └── utils.py                # safe_norm, soft_thresholding, solve_cubic
├── score/                      # Importance score computation
│   ├── wanda_score.py          # Wanda: ||activation||_2 per weight
│   ├── get_grad.py             # Magnitude, 1st-order, 2nd-order Taylor scores
│   └── score_choos.py          # Score selection interface
├── scheduler/
│   └── C_Sche.py               # Sine-based schedule for C
├── tests/                      # Unit & smoke tests
│   ├── test_pruning.py
│   ├── test_ppercent_smoke.py
│   └── test_percentile_consistency.py
├── results/
│   ├── EXPERIMENT_REPORT.md    # Full experiment report with tables
│   ├── metrics/                # JSON experiment logs
│   ├── sweeps/                 # CSV sweep outputs
│   └── figures/, plots/        # Visualization outputs
├── run_full_experiment.py      # Main experiment driver (all 360 configs)
├── run_sweep_admm.py           # ADMM-only sweep script
├── run_sweep_full.py           # Full sweep across all methods
├── tune_hyperparameters.py     # Hyperparameter tuning
├── debug_convergence.py        # Sanity check: loss ↓ & sparsity ↑ over 50 steps
├── analyze_results.py          # Result analysis & table generation
├── generate_plots.py           # Accuracy-sparsity curve plotting
└── review.md                   # NeurIPS 2026 positioning & literature review
```

---

## Score Types

| Score | Formula | Requires | Stability |
|-------|---------|----------|-----------|
| **Magnitude** | $|W|$ | Nothing (data-free) | ★★★★★ |
| **First-Order** | $|(\partial L / \partial W) \cdot W|$ | 1 backward pass | ★★★☆☆ |
| **Second-Order** | $\frac{1}{2}\sum (\partial L / \partial W \cdot W)^2$ | Multiple backward passes | ★★★★☆ |
| **First+Second** | Combined | Multiple backward passes | ★★★★☆ |

---

## Scope Comparison

| Scope | Granularity | Best For | Caveat |
|-------|-------------|----------|--------|
| **Global** | All weights as one group | Extreme compression (>98%) | Less fine-grained control |
| **Layer** | Per-layer groups | Moderate pruning | Can collapse at high C (≥0.08) |
| **Neuron** | Per output-neuron groups | Best accuracy retention | Higher compute cost |

---

## Quick Start

```python
import torch
from network.cnn3 import CNN
from score.wanda_score import WANDA_ScoreCalculator
from score.get_grad import GradientCollector
from score.score_choos import choose_score
from optimizer.ADMM_global import ADMM_Adam_global

model = CNN()
# 1) Collect activations via forward pass
wanda_calc = WANDA_ScoreCalculator(model)
x = torch.randn(8, 1, 28, 28)
_ = model(x)

# 2) Compute importance scores
collector = GradientCollector(model)
scores = choose_score(wanda_calc, collector, "second order")

# 3) Initialize ADMM buffers
params = list(model.parameters())
zeros_like = [torch.zeros_like(p) for p in params]
optimizer = ADMM_Adam_global(params, lr=0.002, N=60000, C=0.03,
                             vk=zeros_like, wk=zeros_like,
                             yk=zeros_like, zk=zeros_like,
                             score=list(scores.values()))

# 4) Training loop
loss = model(x).sum()
loss.backward()
optimizer.step()
```

---

## Running Experiments

### Sanity Check (run first!)
```bash
python debug_convergence.py
# Success: loss ↓ and sparsity ↑ over 50 steps. Fail: NaN loss or 0% sparsity.
```

### Full Experiment (360 configurations)
```bash
python run_full_experiment.py
# Outputs: results/metrics/full_experiment_<timestamp>.json
```

### Targeted Sweeps
```bash
# ADMM-only sweep
python run_sweep_admm.py --c-values 0.5 1.0 --lr-values 0.001 0.0005 \
    --train-steps 120 --epochs 1 --val-size 1000 --outfile results/sweeps/admm_sweep.csv

# All 9 optimizers × multiple scores
python run_sweep_full.py \
    --optimizers ADMM_global ADMM_layer ADMM_neuron \
                 Ppercent_global Ppercent_layer Ppercent_neuron \
                 Lasso_global Lasso_layer Lasso_neuron \
    --score-names "first order" "second order" "first order + second order" \
    --c-values 0.5 1.0 --p-values 20 40 --lr-values 0.001 0.0005 \
    --train-steps 200 --epochs 1 --val-size 1000 \
    --outfile results/sweeps/full_sweep.csv
```

### Hyperparameter Tuning
```bash
python tune_hyperparameters.py
```

---

## Key Findings

1. **ADMM achieves extreme compression**: 100× compression (99% sparsity) at 94.66% accuracy via ADMM_neuron.
2. **Neuron-wise scope outperforms others**: Consistently best accuracy among ADMM variants at equivalent sparsity.
3. **Magnitude scores are most reliable**: Data-free, stable across all methods — gradient-based scores offer marginal improvements.
4. **Optimal learning rate is 0.002**: Doubling the default (0.001) significantly improves all ADMM variants.
5. **Lasso_neuron is uniquely robust**: Accuracy stays 96.7–96.95% across C ∈ [0.01, 10.0] with ~77% sparsity.
6. **Sharp phase transition in ADMM**: Both ADMM_layer and ADMM_neuron collapse (accuracy → 9.8%) when C exceeds a critical threshold (~0.08).

---

## Practical Recommendations

| Goal | Config |
|------|--------|
| Maximum Accuracy | `Ppercent_global`, second-order, p=10 |
| Balanced Performance | `ADMM_neuron`, magnitude, C=0.02 |
| High Compression (>90%) | `ADMM_neuron`, magnitude, C=0.03 |
| Extreme Compression (>98%) | `ADMM_global`, magnitude, C=0.03 |
| Hyperparameter Robustness | `Lasso_neuron`, magnitude, any C |

---

## Implementation Notes

- All ADMM variants expect `score` tensors ordered identically to `model.parameters()`.
- `safe_norm` adds ε=1e-8 to avoid division by zero; `safe_cbrt` handles negative cube roots via `sign(x)·|x|^{1/3}`.
- Soft-thresholding thresholds are clamped to `[1e-6, 0.1]` to prevent over-pruning.
- The cubic solver uses Cardano's formula (one real root) with trigonometric fallback (three real roots) and enforces τ ≥ 1.
- Random seed: 42 for reproducibility.

---

## Future Work

1. **Scale to larger datasets/models**: CIFAR-10, ImageNet, Vision Transformers
2. **LLM pruning**: Apply to Llama-3.1, OPT — compare against SparseGPT, Wanda, SPAP
3. **Structured pruning**: Extend to channel/filter-level for hardware speedups
4. **Dynamic sparsity scheduling**: Principled C-annealing based on gradient SNR
5. **Convergence theory**: Extend stochastic ADMM convergence proofs to the Ratio Norm case

---

## References

- **Wanda**: Sun et al., 2023 — Activation-aware weight pruning
- **SparseGPT**: Frantar & Alistarh, 2023 — Hessian-based LLM pruning
- **STRUPRUNE**: 2025 — ADMM-based structured pruning
- **SPAP/FASP**: 2025 — Alternating penalty pruning for LLMs
- **L1/L2 Ratio Norm**: Scale-invariant sparsity via Hoyer measure

*Project status: Sanity check & MNIST validation complete. Preparing for large-scale experiments (NeurIPS 2026 target).*
