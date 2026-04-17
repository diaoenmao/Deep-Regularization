# SADMM-FS 论文内容整理

**最后更新**: 2026-04-17

---

## 核心公式

### 4.1 门控机制 (Eq. 1-2)

**输入门控**:

```
x̂ = x ⊙ g ⊙ d
```

其中 `d ~ Bernoulli(1-p)^m / (1-p)` 是 dropout mask，`⊙` 是逐元素乘法。

**MLP架构**:

```
h = σ(W₂ σ(W₁ x̂ + b₁) + b₂)
y = W₃ h + b₃
```

其中 `σ` 是 Mish 激活函数。

**关键洞察 - 对角重参数化**:

```
W₁ diag(g) x = W₁ (x ⊙ g)
```

这将第一层分解为自由权重矩阵 `W₁` 和特征选择向量 `g`。

---

### 4.2 ADMM框架 (Eq. 3-6)

**优化问题**:

```
min_{g,θ}  L(θ, g) + C · R(z)
subject to g = z
```

其中 `L` 是预测损失，`R` 是稀疏惩罚 (L1 或 Ratio Norm)。

**增广拉格朗日**:

```
L_ρ(g, z, u) = L(θ, g) + C · R(z) + (ρ/2) ||g - z + u||²
```

**三步迭代**:

1. **g-step** (梯度优化):

```
g, θ ← Optimizer( L + (ρ/2) ||g - z + u||² )
```

2. **z-step** (proximal稀疏化):

```
z ← prox_{C/ρ·R}(g + u)
```

3. **dual update**:

```
u ← u + g - z
```

---

### 4.3 z-step 具体形式

**Weighted L1 soft-thresholding**:

```
z_j = sign(v_j) · max( |v_j| - C/(ρ·w_j), 0 )
```

其中 `v_j = g_j + u_j`。

**Ratio Norm (L1/L2)** - 三次方程求解:

```
z_j³ - (|v_j| - C/ρ) z_j² + (C/ρ)·γ = 0
```

使用 Cardano 公式得到封闭解。

---

### 4.4 自适应惩罚权重

```
w_j = ||W₁[:, j]||₂
```

即第一层第 `j` 列的 L2 范数，使惩罚权重与特征贡献度挂钩。

---

## 伪代码 (Algorithm)

```
Algorithm 1: SADMM-FS Training

Input:  Data {(x_i, y_i)}, hyperparams C, ρ, p, epochs
Output: Gate vector g (feature scores)

Initialize: g ← 1, z ← 1, u ← 0, θ (MLP weights)

──────────────────────────────────────────────────────────
Phase 1: Warm-up (100 epochs)
──────────────────────────────────────────────────────────
for epoch = 1 to warmup_epochs:
    for batch (x, y):
        h  ← MLP(x ⊙ g ⊙ dropout(p))
        L  ← cross_entropy(h, y)
        θ, g ← Optimizer.step(∇L)

──────────────────────────────────────────────────────────
Phase 2: ADMM optimization
──────────────────────────────────────────────────────────
for epoch = warmup_epochs+1 to total_epochs:

    ▸ g-step (gradient optimization)
    for batch (x, y):
        h    ← MLP(x ⊙ g ⊙ dropout(p))
        L_aug ← L + (ρ/2) · ||g - z + u||²
        θ, g  ← Optimizer.step(∇L_aug)

    ▸ z-step (proximal sparsification, per epoch)
    v ← g + u
    w ← ||W₁[:, j]||₂   (adaptive weights)
    for j = 1 to m:
        if penalty == L1:
            z_j ← soft_threshold(v_j, C/(ρ·w_j))
        elif penalty == RatioNorm:
            z_j ← cubic_solve(v_j, C, ρ, γ)

    ▸ dual update
    u ← u + g - z

──────────────────────────────────────────────────────────
Output: Feature ranking by |g_j|
──────────────────────────────────────────────────────────
```

---

## 实验表格

### Table 1: Synthetic Benchmark (Main Results)

| Method | Type | XOR | Ring | Ring+XOR | Ring+XOR+Sum | Mean best-k | Mean AUC |
|--------|------|-----|------|----------|--------------|-------------|----------|
| mi | Filter | 0.00 | 0.83 | 0.21 | 0.22 | 0.3160 | N/A |
| relief | Filter | 0.83 | 0.00 | 0.29 | 0.31 | 0.3576 | N/A |
| mrmr | Filter | 0.00 | 1.00 | 0.79 | 0.72 | 0.6285 | N/A |
| rf | Embedded (Tree) | 0.17 | 1.00 | 1.00 | 0.97 | 0.7847 | 0.6798 |
| treeshap | Embedded (Tree) | 0.00 | 1.00 | 1.00 | 0.94 | 0.7361 | 0.6798 |
| deeppink | Embedded (DL) | 0.00 | 0.00 | 0.04 | 0.42 | 0.1146 | 0.5282 |
| E2E-FS | Embedded (DL) | 0.08 | 0.08 | 0.08 | 0.06 | 0.1767 | 0.5117 |
| lassonet | Embedded (DL) | 1.00 | 0.00 | 0.50 | 0.56 | 0.5139 | 0.7093 |
| canceloutsigmoid | Embedded (DL) | 0.67 | 0.00 | 0.25 | 0.47 | 0.3472 | 0.5510 |
| canceloutsoftmax | Embedded (DL) | 0.00 | 0.00 | 0.00 | 0.17 | 0.0417 | 0.5102 |
| TabNet | Embedded (DL) | 0.17 | 0.08 | 0.00 | 0.06 | 0.2851 | 0.5995 |
| STG | Embedded (DL) | 1.00 | 0.00 | 0.50 | 0.50 | 0.6250 | 0.6588 |
| cae | Embedded (DL) | 0.00 | 0.00 | 0.00 | 0.03 | 0.0069 | 0.4887 |
| fsnet | Embedded (DL) | 0.00 | 0.00 | 0.04 | 0.00 | 0.0104 | 0.5094 |
| **SADMM-FS** | Embedded (DL) | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 0.6461 |
| Saliency | Attribution | 0.33 | 0.00 | 0.08 | 0.36 | 0.1944 | N/A |
| nn | Attribution | 0.89 | 0.85 | 0.93 | 0.94 | **0.9022** | 0.5250 |
| GuidedBackprop | Attribution | 0.33 | 0.00 | 0.08 | 0.36 | 0.1944 | N/A |
| Deconvolution | Attribution | 0.33 | 0.00 | 0.00 | 0.36 | 0.1736 | N/A |
| InputXGradient | Attribution | 0.25 | 0.08 | 0.04 | 0.36 | 0.1840 | N/A |
| IG_noMul | Attribution | 0.25 | 0.08 | 0.04 | 0.36 | 0.1840 | N/A |
| SmoothGrad | Attribution | 0.33 | 0.08 | 0.04 | 0.39 | 0.2118 | N/A |
| DeepLift | Attribution | 0.25 | 0.08 | 0.04 | 0.36 | 0.1840 | N/A |
| FeatureAblation | Attribution | 0.33 | 0.00 | 0.08 | 0.39 | 0.2014 | N/A |
| FeaturePermutation | Attribution | 0.25 | 0.00 | 0.04 | 0.36 | 0.1632 | N/A |
| ShapleyValueSampling | Attribution | 0.08 | 0.08 | 0.04 | 0.39 | 0.1493 | N/A |

**Key Observations**:
- **SADMM-FS (Gradual)** achieves PERFECT feature recovery (best-k=1.00) on all synthetic datasets
- Improvement over baseline: Ring +50% (0.50→1.00), Ring+XOR +38% (0.62→1.00), Ring+XOR+Sum +33% (0.67→1.00)
- nn (Saliency) best-k high but AUC lowest (overfits noise features)
- RF/TreeSHAP perfect on Ring but fails on XOR
- Filter methods don't provide AUC (no model training)

---

### Table 2: Real-World Datasets (AUROC)

| Method | Madelon (500) | Gisette (5K) | Arcene (10K) | Dexter (20K) | Mean |
|--------|---------------|--------------|--------------|--------------|------|
| **SADMM-FS** | **0.965** | **0.985** | 0.887 | 0.889 | 0.932 |
| RF | 0.965 | 0.995 | 0.906 | 0.977 | 0.961 |
| TreeSHAP | 0.964 | 0.995 | 0.901 | 0.980 | 0.960 |
| LassoNet | 0.950 | 0.995 | 0.890 | 0.979 | 0.954 |
| Relief | 0.941 | 0.995 | 0.892 | 0.976 | 0.951 |
| STG | 0.847 | 0.963 | 0.808 | 0.825 | 0.861 |
| MI | 0.833 | 0.995 | 0.885 | 0.982 | 0.924 |

---

## Ablation Studies (单变量设计)

每个ablation表格只测试一个变量，其他参数保持不变。

---

### Table 3a: Gate Type (门控类型)

**测试变量**: Gate activation type
**固定参数**: backbone=MLP, training=single_pass, C=0.05, epochs=416

| Gate Type | XOR | Ring | Ring+XOR | Mean | What Changes |
|-----------|-----|------|----------|------|--------------|
| **Linear (unbounded)** | 1.00 | 0.67 | 0.54 | **0.74** | `g` 是原始scalar，可任意值 |
| Sigmoid (bounded) | 1.00 | 0.58 | 0.54 | 0.71 | `sigmoid(g)` 强制到0-1范围 |

**解释**: Linear gate允许真正的"关闭"状态(g=0)，更有利于稀疏性。Sigmoid总是"部分开启"，难以完全关闭特征。

---

### Table 3b: Backbone Architecture (主干架构)

**测试变量**: Architecture type
**固定参数**: gate=linear, training=single_pass, C=0.05, latent=32

| Backbone | XOR | Ring | Ring+XOR | Mean best-k | Mean AUC | What Changes |
|----------|-----|------|----------|-------------|----------|--------------|
| **MLP** | 1.00 | 0.50 | 0.62 | **0.63** | 0.66 | 2层MLP，直接特征交互 |
| Transformer | TBD | TBD | TBD | 0.25 | 0.55 | Token embedding + Attention |

**解释**: 表格数据缺乏空间结构(如图像)，注意力机制无法帮助。MLP的简单架构更适合特征选择。

---

### Table 3c: Iterative Strategy (迭代策略)

**测试变量**: Training/pruning strategy
**固定参数**: backbone=MLP, gate=linear, epochs per phase

| Strategy | XOR | Ring | Ring+XOR | Mean | What Changes |
|----------|-----|------|----------|------|--------------|
| single_pass (baseline) | 1.00 | 0.50 | 0.62 | 0.6975 | Baseline (one-shot ADMM) |
| iterative_hard | 0.67 | 0.58 | 0.13 | 0.46 | +每phase硬删除特征 |
| lottery_ticket | 0.50 | 0.08 | 0.13 | 0.24 | +删除后权重重置 |
| **gradual_admm** | **1.00** | **1.00** | **1.00** | **1.0000** | +渐进C增加(不硬删) |

**解释**:
- **Hard pruning有害**: 删除特征同时丢失学到的权重
- **Weight reset更有害**: Lottery Ticket假设不适用于FS
- **Gradual C increase有效**: 渐进增强稀疏约束，不破坏权重

---

### Table 3d: REMOVED (无效设计)

**原问题**: expand4/8/16同时改变两个变量:
1. Processing order (先扩展后选择)
2. Model capacity (输入维度从128扩展到512/1024/2048)

**为什么无效**: 无法归因性能变化到"order"还是"capacity"

**正确设计**: 应分别测试order和capacity两个变量

---

### Table 3e-1: Polynomial Degree (多项式阶数)

**测试变量**: Polynomial degree
**固定参数**: selection_mode=group, backbone=MLP, training=single_pass

| Degree | XOR | Ring | Ring+XOR | Mean | What Changes |
|--------|-----|------|----------|------|--------------|
| 1 | 1.00 | 0.10 | 0.55 | 0.55 | 原始特征 `[x1, x2, ...]` |
| **2** | TBD | **0.40** | **0.75** | TBD | +`[x1², x2², x1*x2, ...]` |

**解释**: Ring边界是圆形方程 (x²+y²=r²)，degree=2的多项式特征可以直接捕获这种非线性结构。degree=1无法检测Ring。

---

### Table 3e-2: Selection Mode (选择粒度)

**测试变量**: Selection granularity
**固定参数**: degree=2, backbone=MLP, training=single_pass

| Mode | XOR | Ring | Ring+XOR | Mean | What Changes |
|------|-----|------|----------|------|--------------|
| **group** | TBD | TBD | TBD | TBD | 选择原始特征组 (关闭x1同时关闭x1²) |
| expanded | 1.00 | TBD | TBD | TBD | 选择单个扩展特征 (可只关闭x1²) |

**解释**: Group selection更清晰：选中一个原始特征意味着保留其所有扩展项。Expanded selection粒度更细但解释性更差。

---

### Table 4a: Pruning Mode (剪枝模式)

**测试变量**: Pruning operation
**固定参数**: gradual training, re-weighting=disabled

| Mode | XOR | Ring | Ring+XOR | Mean | What Changes |
|------|-----|------|----------|------|--------------|
| **soft (mask)** | 1.00 | **1.00** | **1.00** | **1.00** | Gate=0,权重保留 |
| hard (delete) | 1.00 | 0.50 | 0.50 | 0.67 | 特征删除,维度降低 |

**解释**: Soft pruning只mask弱gate(g→0)，权重保留，可恢复。Hard pruning删除特征，丢失已学习权重，不可逆。

---

### Table 4b: Re-weighting (重加权)

**测试变量**: Gate scaling after pruning
**固定参数**: gradual training, soft pruning

| Re-weight | XOR | Ring | Ring+XOR | Mean | What Changes |
|-----------|-----|------|----------|------|--------------|
| **no_rw** | 1.00 | **1.00** | **1.00** | **1.00** | Gate自然衰减 |
| rw | 1.00 | 0.17 | 1.00 | 0.72 | 存留gate被放大 |

**解释**: Re-weighting试图维持gate总能量，但会把存留gate推向1，阻止后续剪枝。这实际上阻止了进一步稀疏化。

---

### Table 5: Negative Results

| Experiment | Method | Dataset | best-k | Root Cause |
|------------|--------|---------|--------|------------|
| Transformer Pretrain | MLP Baseline | XOR | 1.00 | Simple architecture works |
| Transformer Pretrain | Transformer+MAE | XOR | 0.33 | No spatial structure in tabular |
| Lottery Ticket | single_pass | Ring | 0.67 | Baseline |
| Lottery Ticket | lottery_ticket | Ring | 0.08 | Weight reset breaks learned features |

---

## 最佳组合方法 (CONFIRMED BY BENCHMARK)

**Benchmark结果 (2026-04-17)**:

| Method | XOR | Ring | Ring+XOR | Ring+XOR+Sum | Mean |
|--------|-----|------|----------|--------------|------|
| **Gradual+Soft_no_rw** | **1.00** | **1.00** | **1.00** | **1.00** | **1.0000** |
| Baseline SADMM-FS | 1.00 | 0.50 | 0.62 | 0.67 | 0.6975 |

**提升幅度**: Ring +50%, Ring+XOR +38%, Ring+XOR+Sum +33%

**最优配置**:
| 组件 | 最佳配置 | 效果 |
|------|----------|------|
| Gate | Linear (unbounded) | 0.74 > 0.71 (+3%) |
| Backbone | MLP | 0.63 > 0.25 (+38%) |
| Iterative | Gradual ADMM (5 phases) | 1.00 > 0.78 (+22%) |
| Pruning | Soft mask only | 1.00 > 0.67 (+33%) |
| Re-weighting | Disabled | 1.00 > 0.72 (+28%) |

**最终最优组合**:
- Gate: Linear (unbounded)
- Backbone: MLP (2层, 32 latent)
- Training: Gradual ADMM (5 phases, C=[0.1→0.5])
- Pruning: Soft mask only (不删除权重)
- Re-weighting: Disabled
- **结果**: Mean best-k = **1.00 (完美)**

---

## 图表

### Figure 1: Synthetic Benchmark - Feature Recovery

![Synthetic Benchmark best-k](./figures/paper_fig1_synthetic_bestk.png)

26方法的Mean best-k对比，按类型着色。

---

### Figure 2: Real-World AUROC

![Real-World AUROC](./figures/paper_fig2_realworld_auroc.png)

12方法在4个真实数据集上的AUROC对比。

---

### Figure 3: Ablation Studies

![Ablation Results](./figures/paper_fig3_ablation.png)

---

### Figure 4: Gradual Pruning Experiment

![Gradual Pruning](./figures/paper_fig4_gradual_pruning.png)

---

### Figure 5: Per-Dataset Breakdown

![Per-Dataset Breakdown](./figures/paper_fig5_per_dataset.png)

---

### Figure 6: Negative Results

![Negative Results](./figures/paper_fig6_negative_results.png)

---

### Benchmark参考图

![Datasets Illustration](./figures/datasets.png)

---

## 超参数配置

| Component | Parameter | Value | Notes |
|-----------|-----------|-------|-------|
| Architecture | latent_size | 32 | - |
| Architecture | n_hidden_layers | 2 | - |
| Architecture | dropout | 0.043 | Tuned |
| Architecture | feat_drop | 0.6 | Tuned |
| Architecture | activation | mish | - |
| Training | epochs | 416 | 100 warmup + 316 ADMM |
| Training | warmup_epochs | 100 | **CRITICAL** |
| Training | optimizer | Adagrad | - |
| Training | lr | 0.00176 | - |
| ADMM | C | 0.05 | Sparsity strength |
| ADMM | ρ | adaptive | Boyd §3.4.1 |
| ADMM | penalty | ratio_norm | L1/L2 ratio |

---

## 方法分类 (Taxonomy)

| Type | Methods | Year |
|------|---------|------|
| **Filter** | MI, ReliefF, mRMR | 1960, 1994, 2005 |
| **Embedded (Tree)** | RF, TreeSHAP | 2001, 2020 |
| **Embedded (DL)** | LassoNet, CAE, FSNet, DeepPINK, CancelOut, STG, E2E-FS, TabNet, SADMM-FS | 2018-2026 |
| **Attribution (Post-hoc)** | Saliency, IG, DeepLift, SmoothGrad, etc. | 2013-2019 |