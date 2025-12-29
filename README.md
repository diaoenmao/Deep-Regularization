# NEW_Pruning_20251110 (refactored overview)

Minimal notes on how the pruning pieces fit together after the refactor.

## Components
- `network/cnn3.py`: simple MNIST CNN backbone used for experiments.
- `Score/`:
  - `wanda_score.py`: collects layer activations and computes Wanda scores.
  - `get_grad.py`: collects gradients or magnitudes of parameters.
  - `score_choos.py`: single entry `choose_score` to select score strategy.
- `optimizer/`:
  - `utils.py`: shared `soft_thresholding` and `safe_norm` helpers.
  - `ADMM_*`: ADMM-based pruning (global, layer, neuron level) using Wanda scaling.
  - `lasso_*`: (sparse) group lasso style shrinkage.
  - `ppercent_*`: percentile-based hard pruning.
- `scheduler/C_Sche.py`: smooth sine schedule for hyperparameter `C`.

> **Note:** `lasso_neuron` directly uses the fixed implementation; `ppercent_neuron`
> now inlines the fixed implementation (p = percentage to prune). Import as usual.

## Typical usage sketch
```python
import torch
from network.cnn3 import CNN
from Score.wanda_score import WANDA_ScoreCalculator
from Score.get_grad import GradientCollector
from Score.score_choos import choose_score
from optimizer.ADMM_global import ADMM_Adam_global

model = CNN()
# 1) Run a forward pass with a small batch to collect activations
wanda_calc = WANDA_ScoreCalculator(model)
x = torch.randn(8, 1, 28, 28)
_ = model(x)

# 2) Pick scores
collector = GradientCollector(model)
scores = choose_score(wanda_calc, collector, "second order")

# 3) Prepare ADMM buffers matching parameters
params = list(model.parameters())
zeros_like = [torch.zeros_like(p) for p in params]
optimizer = ADMM_Adam_global(params, lr=1e-3, N=60000, C=1.0,
                             vk=zeros_like, wk=zeros_like,
                             yk=zeros_like, zk=zeros_like,
                             score=list(scores.values()))

# 4) Standard training loop
loss = model(x).sum()
loss.backward()
optimizer.step()
```

## Quick sweeps
- ADMM-only tiny sweep:
  - `python run_sweep_admm.py --c-values 0.5 1.0 --lr-values 0.001 0.0005 --train-steps 120 --epochs 1 --val-size 1000 --outfile results/sweeps/admm_sweep.csv`

- Broader sweep (ADMM_global/layer/neuron + Ppercent_global/layer/neuron + Lasso_global/layer/neuron, multiple scores):
  - `python run_sweep_full.py --optimizers ADMM_global ADMM_layer ADMM_neuron Ppercent_global Ppercent_layer Ppercent_neuron Lasso_global Lasso_layer Lasso_neuron --score-names "first order" "second order" "first order + second order" --c-values 0.5 1.0 --p-values 20 40 --lr-values 0.001 0.0005 --train-steps 200 --epochs 1 --val-size 1000 --outfile results/sweeps/full_sweep.csv`

## Notes
- All ADMM variants expect `score` tensors ordered identically to model parameters.
- `safe_norm` adds a small epsilon to avoid division by zero in ADMM updates.
- Percentile and lasso optimizers now clamp thresholds to avoid over-pruning.
