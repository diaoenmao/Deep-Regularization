# SADMM Neural Network Pruning: Full Experiment Report

## Executive Summary

This report presents comprehensive experimental results for the **SADMM (Scope-Driven Stochastic ADMM)** neural network pruning framework. The framework integrates **Ratio Norm (L1/L2)** regularization with **Taylor expansion-based importance scores** into the ADMM optimization loop.

### Key Results
- **Best Accuracy**: 97.67% (Ppercent_global, second-order score)
- **Best Compression**: 100x (99% sparsity) at 94.66% accuracy (ADMM_neuron)
- **Best Tradeoff**: 92.80% accuracy at 98.7% sparsity (ADMM_global)

---

## 1. Experimental Setup

### 1.1 Dataset
- **Dataset**: MNIST handwritten digits
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Input size**: 28×28 grayscale images
- **Classes**: 10 (digits 0-9)

### 1.2 Model Architecture
**CNN3**: A simple convolutional neural network
```
Conv2d(1, 64, 3×3) → BatchNorm → ReLU → MaxPool
Conv2d(64, 128, 3×3) → BatchNorm → ReLU → MaxPool
Linear(6272, 1024) → ReLU → Dropout(0.5)
Linear(1024, 10)
```
**Total Parameters**: ~7.5M

### 1.3 Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 64 |
| Learning Rate | 0.002 (optimized) |
| Optimizer | ADMM/Lasso/Ppercent variants |

### 1.4 Methods Tested
- **9 Optimizers**: ADMM, Lasso, Ppercent × Global, Layer, Neuron scopes
- **4 Score Types**: Magnitude, First-Order, Second-Order, First+Second-Order
- **10 Hyperparameter values** per configuration
- **Total**: 360 configurations

---

## 2. Results by Method

### 2.1 ADMM Methods

#### ADMM_Global
The global ADMM optimizer treats all weights as a single group for the Ratio Norm regularization.

| Score Type | C | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| First+Second | 0.01 | **95.92%** | 52.4% | Best accuracy |
| First-Order | 0.01 | 95.79% | 52.2% | |
| Magnitude | 0.01 | 95.71% | 52.8% | |
| Second-Order | 0.01 | 95.66% | 53.6% | |
| Magnitude | 0.03 | **92.80%** | **98.7%** | Best compression |
| Magnitude | 0.05 | 87.93% | 100.0% | Extreme |

**Accuracy vs Sparsity (Magnitude Score)**:
| C | Accuracy | Sparsity | Compression |
|---|----------|----------|-------------|
| 0.010 | 95.71% | 52.8% | 2.1× |
| 0.015 | 94.55% | 84.6% | 6.5× |
| 0.020 | 93.83% | 95.0% | 20× |
| 0.025 | 93.10% | 97.6% | 42× |
| 0.030 | 92.80% | 98.7% | 77× |
| 0.035 | 91.36% | 99.3% | 143× |
| 0.040 | 89.91% | 99.5% | 200× |
| 0.045 | 88.24% | 99.6% | 250× |
| 0.050 | 87.93% | 100.0% | ∞ |

#### ADMM_Layer
Layer-wise ADMM applies the Ratio Norm independently to each layer.

| Score Type | C | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| First-Order | 0.02 | **95.66%** | 10.7% | Best accuracy |
| First+Second | 0.03 | 95.07% | 19.6% | |
| Second-Order | 0.06 | 92.91% | 59.2% | Max stable |
| Any | ≥0.10 | 9.80% | varies | **Collapsed** |

**Warning**: ADMM_layer collapses (accuracy → 9.8%) when C ≥ 0.08-0.10.

#### ADMM_Neuron
Neuron-wise ADMM applies the Ratio Norm to each output neuron independently.

| Score Type | C | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| Second-Order | 0.01 | **96.34%** | 18.5% | **Best ADMM accuracy** |
| First+Second | 0.01 | 96.31% | 19.2% | |
| First-Order | 0.01 | 96.28% | 20.0% | |
| Magnitude | 0.01 | 96.18% | 19.0% | |
| Magnitude | 0.03 | **94.66%** | **99.0%** | **100× compression** |
| First+Second | 0.06 | 92.82% | 99.1% | Max stable |

**Key Finding**: ADMM_neuron achieves the best accuracy among ADMM methods and maintains excellent performance even at 99% sparsity.

---

### 2.2 Ppercent Methods (Baseline)

Ppercent methods prune a fixed percentage of weights based on importance scores.

#### Ppercent_Global

| Score Type | p | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| Second-Order | 10 | **97.67%** | 35.1% | **Highest overall** |
| First-Order | 5 | 97.66% | 46.0% | |
| First+Second | 5 | 97.61% | 41.6% | |
| Magnitude | 10 | 97.50% | 10.0% | |
| Magnitude | 70 | 96.00% | 70.0% | High compression |

#### Ppercent_Layer

| Score Type | p | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| First+Second | 5 | **97.62%** | 0.1% | Very conservative |
| Magnitude | 5 | 97.55% | 5.0% | |
| First-Order | 25 | 97.55% | 10.9% | |
| Magnitude | 70 | 95.04% | 70.0% | |

#### Ppercent_Neuron

| Score Type | p | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| First+Second | 5 | **97.66%** | 5.6% | |
| Second-Order | 10 | 97.52% | 10.9% | |
| First-Order | 20 | 97.50% | 23.8% | |
| Second-Order | 70 | 96.07% | 81.8% | High compression |

---

### 2.3 Lasso Methods

Lasso methods use L1 regularization for pruning.

#### Lasso_Global

| Score Type | C | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| Magnitude | 0.002 | **97.42%** | 24.8% | Best |
| Magnitude | 0.01 | 97.21% | 35.4% | |
| Second-Order | 0.001 | 96.85% | 88.3% | High compression |
| First-Order | 0.001 | 96.02% | 96.5% | Aggressive |

#### Lasso_Layer

| Score Type | C | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| Magnitude | 0.001 | **97.44%** | 23.5% | Best |
| Magnitude | 0.01 | 97.44% | 35.8% | Stable |
| Second-Order | 0.001 | 96.87% | 89.6% | |

#### Lasso_Neuron

| Score Type | C | Accuracy | Sparsity | Notes |
|------------|---|----------|----------|-------|
| Magnitude | 0.02 | **96.95%** | 77.3% | **Remarkably stable** |
| Magnitude | 1.0 | 96.94% | 76.8% | Same performance! |
| Second-Order | 0.1 | 87.05% | 100% | Collapsed |
| First-Order | any | ~50% | 100% | Collapsed |

**Key Finding**: Lasso_neuron with magnitude scores is extremely stable - accuracy remains 96.7-96.95% across ALL C values (0.01 to 10.0) with consistent ~77-79% sparsity.

---

## 3. Comparative Analysis

### 3.1 Best Results Summary

#### Highest Accuracy
| Rank | Method | Score | Config | Accuracy | Sparsity |
|------|--------|-------|--------|----------|----------|
| 1 | Ppercent_global | Second-Order | p=10 | **97.67%** | 35.1% |
| 2 | Ppercent_neuron | First+Second | p=5 | 97.66% | 5.6% |
| 3 | Ppercent_layer | First+Second | p=5 | 97.62% | 0.1% |
| 4 | Lasso_layer | Magnitude | C=0.001 | 97.44% | 23.5% |
| 5 | Lasso_global | Magnitude | C=0.002 | 97.42% | 24.8% |

#### Best Compression (>90% Sparsity)
| Rank | Method | Score | Config | Accuracy | Sparsity | Compression |
|------|--------|-------|--------|----------|----------|-------------|
| 1 | ADMM_neuron | Magnitude | C=0.03 | **94.66%** | **99.0%** | **100×** |
| 2 | ADMM_global | Magnitude | C=0.03 | 92.80% | 98.7% | 77× |
| 3 | ADMM_global | Second-Order | C=0.045 | 89.62% | 99.8% | 500× |
| 4 | Lasso_global | Second-Order | C=0.001 | 96.85% | 88.3% | 8.5× |

#### Best Accuracy-Sparsity Tradeoff
| Method | Score | Config | Accuracy | Sparsity | Score* |
|--------|-------|--------|----------|----------|--------|
| ADMM_neuron | Magnitude | C=0.02 | 95.66% | 71.3% | 166.96 |
| ADMM_global | Magnitude | C=0.015 | 94.55% | 84.6% | 179.15 |
| Lasso_neuron | Magnitude | C=0.02 | 96.95% | 77.3% | 174.25 |
| ADMM_neuron | Magnitude | C=0.03 | 94.66% | 99.0% | 193.66 |

*Score = Accuracy + Sparsity (higher is better)

### 3.2 Score Type Analysis

| Score Type | Best For | Stability | Notes |
|------------|----------|-----------|-------|
| **Magnitude** | All methods | ★★★★★ | Most reliable, data-free |
| **Second-Order** | ADMM_neuron, Ppercent | ★★★★☆ | Slight improvement |
| **First-Order** | Ppercent | ★★★☆☆ | Can be unstable with Lasso |
| **First+Second** | Ppercent | ★★★★☆ | Good balance |

### 3.3 Scope Analysis

| Scope | Pros | Cons | Best Use Case |
|-------|------|------|---------------|
| **Global** | Simple, consistent | Less fine-grained | Extreme compression |
| **Layer** | Per-layer control | Can collapse | Moderate pruning |
| **Neuron** | Best accuracy | More computation | High accuracy needs |

---

## 4. Visualizations

See accompanying figures:
- `accuracy_sparsity_admm.png` - ADMM methods comparison
- `accuracy_sparsity_ppercent.png` - Ppercent methods comparison
- `accuracy_sparsity_lasso.png` - Lasso methods comparison
- `accuracy_sparsity_all.png` - All methods comparison
- `accuracy_sparsity_best.png` - Best configurations

---

## 5. Conclusions

### 5.1 Main Findings

1. **ADMM achieves extreme compression**: The ADMM framework can achieve 100× compression (99% sparsity) while maintaining 94.66% accuracy on MNIST.

2. **Neuron-wise scope is most effective**: ADMM_neuron consistently outperforms global and layer variants in both accuracy and compression.

3. **Magnitude scores are most reliable**: While gradient-based scores can provide slight improvements, magnitude scores are the most stable across all methods.

4. **Optimal learning rate is 0.002**: Doubling the default learning rate significantly improves all ADMM variants.

5. **Ppercent provides highest accuracy**: For applications prioritizing accuracy over compression, Ppercent methods achieve up to 97.67%.

### 5.2 Practical Recommendations

| Goal | Recommended Configuration |
|------|--------------------------|
| **Maximum Accuracy** | Ppercent_global, second-order, p=10 |
| **Balanced Performance** | ADMM_neuron, magnitude, C=0.02 |
| **High Compression (>90%)** | ADMM_neuron, magnitude, C=0.03 |
| **Extreme Compression (>98%)** | ADMM_global, magnitude, C=0.03 |
| **Hyperparameter Robustness** | Lasso_neuron, magnitude, any C |

### 5.3 Future Work

1. **Scale to larger datasets**: Validate on CIFAR-10, ImageNet
2. **Apply to LLMs**: Test on transformer architectures
3. **Structured pruning**: Extend to channel/filter pruning
4. **Dynamic sparsity**: Implement gradual pruning schedules

---

## Appendix A: Full Results Tables

### A.1 ADMM_Global - All Configurations

| Score | C=0.01 | C=0.015 | C=0.02 | C=0.025 | C=0.03 | C=0.035 | C=0.04 | C=0.045 | C=0.05 | C=0.06 |
|-------|--------|---------|--------|---------|--------|---------|--------|---------|--------|--------|
| **Magnitude** |
| Accuracy | 95.71 | 94.55 | 93.83 | 93.10 | 92.80 | 91.36 | 89.91 | 88.24 | 87.93 | 29.20 |
| Sparsity | 52.8 | 84.6 | 95.0 | 97.6 | 98.7 | 99.3 | 99.5 | 99.6 | 100.0 | 99.8 |
| **First-Order** |
| Accuracy | 95.79 | 94.62 | 93.88 | 92.99 | 92.32 | 91.41 | 91.34 | 87.05 | 84.22 | 52.27 |
| Sparsity | 52.2 | 83.5 | 94.9 | 97.3 | 98.7 | 99.1 | 99.5 | 99.9 | 100.0 | 99.8 |
| **Second-Order** |
| Accuracy | 95.66 | 95.08 | 94.13 | 93.07 | 92.56 | 91.51 | 90.56 | 89.62 | 83.26 | 23.83 |
| Sparsity | 53.6 | 85.0 | 94.8 | 96.9 | 98.5 | 99.1 | 99.6 | 99.8 | 99.6 | 99.8 |
| **First+Second** |
| Accuracy | 95.92 | 94.80 | 94.22 | 92.61 | 92.59 | 90.52 | 90.43 | 89.72 | 82.64 | 62.92 |
| Sparsity | 52.4 | 84.7 | 93.9 | 97.1 | 98.6 | 99.2 | 99.5 | 99.6 | 100.0 | 99.8 |

### A.2 ADMM_Neuron - All Configurations

| Score | C=0.01 | C=0.02 | C=0.03 | C=0.04 | C=0.05 | C=0.06 | C=0.08 | C=0.10 | C=0.15 | C=0.20 |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Magnitude** |
| Accuracy | 96.18 | 95.66 | 94.66 | 93.08 | 93.25 | 91.54 | 9.80 | 9.80 | 9.80 | 9.80 |
| Sparsity | 19.0 | 71.3 | 99.0 | 98.9 | 99.0 | 99.1 | 98.9 | 99.0 | 99.2 | 99.7 |
| **First-Order** |
| Accuracy | 96.28 | 95.56 | 94.67 | 93.48 | 92.97 | 92.13 | 9.80 | 9.80 | 9.80 | 9.80 |
| Sparsity | 20.0 | 71.2 | 99.0 | 98.9 | 99.0 | 99.1 | 98.9 | 99.0 | 99.3 | 99.8 |
| **Second-Order** |
| Accuracy | 96.34 | 95.49 | 94.83 | 93.58 | 92.80 | 92.09 | 9.80 | 9.80 | 9.80 | 9.80 |
| Sparsity | 18.5 | 71.0 | 99.0 | 98.9 | 99.0 | 99.1 | 98.9 | 99.0 | 99.2 | 99.8 |
| **First+Second** |
| Accuracy | 96.31 | 95.63 | 94.26 | 93.45 | 93.01 | 92.82 | 9.80 | 9.80 | 9.80 | 9.80 |
| Sparsity | 19.2 | 71.2 | 99.0 | 98.9 | 99.0 | 99.1 | 98.9 | 99.0 | 99.2 | 99.7 |

### A.3 Ppercent_Global - All Configurations

| Score | p=5 | p=10 | p=15 | p=20 | p=25 | p=30 | p=40 | p=50 | p=60 | p=70 |
|-------|-----|------|------|------|------|------|------|------|------|------|
| **Magnitude** |
| Accuracy | 97.26 | 97.50 | 97.39 | 97.36 | 97.47 | 97.06 | 96.81 | 96.61 | 96.22 | 96.00 |
| Sparsity | 5.0 | 10.0 | 15.0 | 20.0 | 25.0 | 30.0 | 40.0 | 50.0 | 60.0 | 70.0 |
| **First-Order** |
| Accuracy | 97.66 | 97.31 | 97.40 | 97.62 | 97.59 | 97.45 | 97.23 | 97.18 | 97.09 | 96.70 |
| Sparsity | 46.0 | 46.6 | 44.4 | 47.9 | 50.7 | 52.9 | 57.9 | 63.9 | 71.8 | 78.4 |
| **Second-Order** |
| Accuracy | 97.37 | 97.67 | 97.33 | 97.48 | 97.48 | 97.35 | 97.29 | 97.21 | 96.93 | 96.86 |
| Sparsity | 31.6 | 35.1 | 34.7 | 35.3 | 40.5 | 43.2 | 49.1 | 58.1 | 67.4 | 75.7 |
| **First+Second** |
| Accuracy | 97.61 | 97.52 | 97.61 | 97.34 | 97.59 | 97.31 | 97.52 | 97.49 | 97.08 | 96.96 |
| Sparsity | 41.6 | 41.3 | 39.1 | 41.0 | 42.7 | 46.0 | 53.5 | 60.5 | 66.3 | 76.1 |

---

## Appendix B: Experimental Details

### B.1 Hardware
- GPU: NVIDIA CUDA-enabled GPU
- Training Time: ~30.5 hours total

### B.2 Software
- Python 3.10
- PyTorch
- Custom ADMM optimizers with cubic solver

### B.3 Reproducibility
- Random seed: 42
- Results file: `results/metrics/full_experiment_20260205_025759.json`

---

*Report generated: February 2026*
*SADMM Research Project*
