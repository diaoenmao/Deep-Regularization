# Fairness Protocol for Neural Feature Selection Comparison

**Version**: 2026-04-03
**Status**: Draft

## Purpose

Define what constitutes a "fair" comparison between neural feature-selection methods, ensuring reproducibility and honest reporting.

---

## 1. Matched Backbone Definition

A **matched backbone** comparison requires:

| Aspect | Requirement |
|--------|-------------|
| Architecture | Same number of layers, same hidden dimensions |
| Activation | Same activation function |
| Initialization | Same weight initialization scheme |
| Input preprocessing | Same normalization/standardization |

**Standard Matched Backbone (Phase 1):**

```python
{
    "latent_size": 32,
    "n_hidden_layers": 2,
    "gaussian_noise": 0.0,
    "dropout": 0.0,
    "activation": "mish"
}
```

---

## 2. Matched Optimizer Definition

A **matched optimizer** comparison requires:

| Aspect | Requirement |
|--------|-------------|
| Optimizer family | Same optimizer (Adam, SGD, Adagrad, etc.) |
| Learning rate | Same initial learning rate |
| Scheduler | Same learning rate schedule |
| Weight decay | Same weight decay coefficient |
| Batch size | Same batch size |

**Standard Matched Training Config:**

```python
{
    "learning_rate": 0.005,
    "epochs": 500,
    "batch_size": 64,
    "weight_decay": 1e-3,
    "optimizer": "adam",
    "early_stopping_patience": 66,
    "warmup_epochs": 120
}
```

**Note**: This matches the main SADMM-FS implementation in `run_admm_input_group()`.
The `nn_wrapper.py` in Feature-Selection-Benchmark uses different defaults (adagrad, lr=0.00176).

---

## 3. Match Level Classification

| Level | Code | Meaning | Example |
|-------|------|---------|---------|
| **Full Match** | `full_match` | Same backbone, same optimizer, same training config | CancelOut, DeepPINK using NNWrapper |
| **Backbone Only** | `backbone_only` | Same architecture, different optimizer/training | STG (Adam vs Adagrad) |
| **Method Specific** | `method_specific` | Same backbone, method requires different training | SADMM-FS with ADMM updates |
| **Standard Implementation** | `standard_impl` | Uses method's canonical implementation | TabNet, FSNet, CAE |

---

## 4. Seed Policy

**For synthetic benchmarks:**

- Use 6-fold cross-validation
- Report mean and standard deviation across folds
- Fixed random seed per fold: `seed = 42 + fold_idx`

**For real-world datasets:**

- Use 5 random seeds
- Report mean and standard deviation
- Seeds: `[42, 123, 456, 789, 1024]`

---

## 5. Metrics

**Primary metrics (report for all methods):**

| Metric | Description | Computation |
|--------|-------------|-------------|
| `best-k` | Fraction of true relevant features in top-k | Higher is better |
| `best-2k` | Fraction in top-2k | Higher is better |
| `AUC` | Area under ROC curve | Higher is better |
| `AUPRC` | Area under PR curve | Higher is better |

**Secondary metrics (optional):**

| Metric | Description |
|--------|-------------|
| `runtime_sec` | Training time per fold |
| `n_parameters` | Model parameter count |

---

## 6. Method Classification

### Tier 1: Neural Feature Selection with Global Gates

| Method | Implementation | Match Level | Notes |
|--------|---------------|-------------|-------|
| SADMM-FS | `admm_input_group_wrapper.py` | method_specific | Main method |
| STG | `stg_wrapper.py` | backbone_only | Adam optimizer |
| CancelOut | `cancelout.py` | full_match | Via NNWrapper |
| DeepPINK | `deeppink.py` | full_match | Knockoff + NNWrapper |

### Tier 2: Encoder-Style Neural FS

| Method | Implementation | Match Level | Notes |
|--------|---------------|-------------|-------|
| FSNet | `fsnet.py` | standard_impl | Separate k/2k models |
| E2E-FS | `e2efs_wrapper.py` | standard_impl | Separate k/2k selectors |
| CAE | `cae_wrapper.py` | standard_impl | Concrete selector |
| TabNet | `tabnet_wrapper.py` | standard_impl | Attentive encoder |

### Tier 3: Traditional Methods

| Method | Implementation | Match Level | Notes |
|--------|---------------|-------------|-------|
| LassoNet | `lassonet` package | standard_impl | Group lasso |
| Lasso | sklearn | N/A | Non-neural |
| RF importance | sklearn | N/A | Non-neural |

---

## 7. Dataset Splits

### Synthetic Datasets

- **Training**: 80% of samples (n=800 for n=1000 grid)
- **Validation**: 20% holdout from training (for early stopping)
- **Test**: 20% of samples (n=200)

```
Total n=1000 → Train=800 → [Fit=640, Val=160], Test=200
```

### Real-World Datasets

- Standard train/test split per dataset
- 5-fold cross-validation within training set
- No data augmentation

---

## 8. Reporting Requirements

### Results JSON Format

```json
{
  "metadata": {
    "timestamp": "YYYYMMDD_HHMMSS",
    "n_samples": 1000,
    "n_folds": 6,
    "matched_backbone": {...},
    "method_name": "SADMM-FS"
  },
  "results": {
    "task_m128": {
      "k": 2,
      "folds": [
        {"best_k": 1.0, "auc": 0.85, "status": "ok"}
      ],
      "mean_best_k": 1.0,
      "std_best_k": 0.0
    }
  },
  "summary": {
    "overall_mean_best_k": 0.81,
    "match_level": "method_specific"
  }
}
```

### Paper Table Format

Include columns for:

| Method | Match Level | Mean best-k | Mean AUC | Training epochs |
|--------|-------------|-------------|----------|-----------------|

---

## 9. Known Disparities

### Training Budget

| Method | Epochs | Notes |
|--------|--------|-------|
| SADMM-FS | 416 | 100 warmup + 316 ADMM |
| STG | 300 | Standard |
| TabNet | 100 | Early stopping |
| FSNet | 2000 | Much longer |

**Impact**: Longer training favors Tier-2 baselines, yet SADMM-FS still outperforms.

### Separate k/2k Models

FSNet, E2E-FS, and CAE train separate models for k and 2k features:

- `best-k` from k-trained model
- `AUC` from 2k-trained model

**Impact**: Gives Tier-2 baselines an advantage (2k features for prediction), yet they still underperform.

---

## 10. Current Status

### Completed Comparisons

- [x] SADMM-FS vs STG (matched backbone)
- [x] SADMM-FS vs CancelOut (full match)
- [x] SADMM-FS vs DeepPINK (full match)
- [x] SADMM-FS vs Tier-2 baselines (standard implementations)
- [x] Real-world datasets (SADMM-FS vs STG)

### Pending

- [ ] LassoNet with matched backbone
- [ ] Ablation: feature dropout on/off
- [ ] Ablation: column-normalized gate variant

---

## References

- Yamada et al. (2020) - STG
- Lemhadri et al. (2021) - LassoNet
- Chang et al. (2019) - CancelOut
- Lu et al. (2018) - DeepPINK
- Arik & Pfister (2021) - TabNet
- Abubakr et al. (2023) - FSNet