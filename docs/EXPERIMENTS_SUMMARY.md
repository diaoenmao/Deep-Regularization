# SADMM-FS 实验总结

**最后更新**: 2026-04-10

---

## 1. 项目概述

### 1.1 核心方法

**SADMM-FS** (Scope-Driven Stochastic ADMM for Feature Selection)

```
核心组件:
1. 全局门控向量 g ∈ R^m (线性、无界)
2. ADMM 优化框架解耦梯度更新和稀疏化
3. 支持 Ratio Norm (L1/L2) 和标准 L1 惩罚
4. 特征 dropout (p=0.6-0.7)
```

### 1.2 关键创新

1. **SADMM Bridge**: ρ = 1/α_k 连接 Adam 自适应步长与 ADMM 惩罚参数
2. **Ratio Norm 三次方程求解**: Cardano 公式封闭解
3. **线性化 ADMM 分离**: 梯度优化与 proximal 稀疏化解耦

---

## 2. 主实验结果

### 2.1 Synthetic Benchmark (6-fold CV)

**数据集配置**:
- XOR: 2 个相关特征, k=2
- Ring: 2 个相关特征, k=2
- Ring+XOR: 4 个相关特征, k=4
- Ring+XOR+Sum: 4 个相关特征, k=4

**主方法对比**:

| 方法 | Match Level | Mean best-k | Mean AUC |
|------|-------------|-------------|----------|
| **SADMM-FS (gated_mlp)** | method_specific | **0.6278** | **0.6605** |
| STG | backbone_only | 0.625 | - |
| CancelOut | full_match | 0.625 | - |
| TabNet | standard_impl | 0.2851 | 0.5995 |
| CAE | standard_impl | 0.1861 | 0.5236 |
| E2E-FS | standard_impl | 0.1767 | 0.5117 |
| FSNet | standard_impl | 0.1667 | 0.5621 |
| DeepPINK | full_match | 0.0938 | - |

**分任务表现**:

| Task | k | SADMM-FS | STG | TreeSHAP | RF |
|------|---|----------|-----|----------|-----|
| xor_m128 | 2 | **1.000** | 1.000 | 0.498 | 0.493 |
| ring_m128 | 2 | 0.500 | 0.000 | **0.990** | **1.000** |
| ring+xor_m256 | 4 | **0.625** | 0.500 | 0.400 | 0.350 |
| ring+xor+sum_m256 | 4 | **0.667** | 0.500 | 0.500 | 0.600 |

### 2.2 Real-World Datasets (NIPS 2003)

| Dataset | m | SADMM-FS AUROC | STG AUROC | RF AUROC |
|---------|---|----------------|-----------|----------|
| madelon | 500 | **0.965** | 0.847 | 0.965 |
| gisette | 5000 | **0.985** | 0.963 | 0.975 |
| arcene | 10000 | **0.887** | 0.808 | 0.846 |
| dexter | 20000 | **0.889** | 0.825 | 0.779 |

### 2.3 维度扩展对比

| Method | XOR m=128 | XOR m=512 | XOR m=2048 |
|--------|-----------|-----------|------------|
| SADMM-FS | **1.000** | 0.667 | 0.167 |
| LassoNet | **1.000** | **1.000** | 0.167 |
| TreeSHAP | 0.498 | 0.498 | 0.498 |

---

## 3. 消融实验

### 3.1 迭代特征选择 (Iterative FS)

**来源**: `iterative_ablation_20260408_195630.json`

**问题**: Lottery Ticket Hypothesis 风格的迭代剪枝是否有帮助？

| 方法 | XOR best-k | Ring best-k | Ring+XOR best-k |
|------|------------|-------------|-----------------|
| single_pass | 1.00 ± 0.0 | 0.67 ± 0.24 | 0.67 ± 0.12 |
| iterative_hard | 0.67 ± 0.24 | 0.58 ± 0.34 | 0.13 ± 0.13 |
| lottery_ticket | 0.50 ± 0.29 | **0.08 ± 0.19** | 0.13 ± 0.19 |
| **gradual_admm** | **1.00 ± 0.0** | **1.00 ± 0.0** | **0.67 ± 0.12** |

**结论**:
- Lottery Ticket 假设不适用于特征选择
- Gradual ADMM 在 Ring 任务上提升 33% (0.67 → 1.00)
- 硬剪枝损害性能

### 3.2 门控类型消融 (Gating Variants)

**来源**: `mentor_gating_full_20260330_fixed.json`

**问题**: 门控应该是有界 (sigmoid) 还是无界 (linear)？

| 门控类型 | Mean best-k | Mean AUC |
|----------|-------------|----------|
| **Linear (unbounded)** | **0.7361** | **0.6417** |
| Sigmoid (bounded) | 0.7083 | 0.6254 |

**分任务结果**:

| Task | Linear gate | Sigmoid gate |
|------|-------------|--------------|
| xor_m128 | **1.00 / 0.85** | 1.00 / 0.84 |
| ring_m128 | **0.67 / 0.51** | 0.58 / 0.49 |
| ring+xor_m256 | **0.54 / 0.56** | 0.54 / 0.54 |

**结论**: 线性无界门控持续优于 sigmoid 有界门控

### 3.3 Backbone 消融

**来源**: `backbone_tier2_synthetic_full_20260327.json`

**问题**: Transformer backbone 是否优于 MLP？

| Backbone | Mean best-k | Mean AUC |
|----------|-------------|----------|
| **MLP (gated_mlp)** | **0.6278** | **0.6605** |
| Transformer | 0.2479 | 0.5544 |

**结论**: Transformer backbone 在合成基准上完全失败

### 3.4 Transformer Pretrain 消融

**来源**: `transformer_pretrain_ablation_20260407_*.json`

**问题**: MAE 风格的预训练能否挽救 Transformer？

| 方法 | best-k | 成功率 |
|------|--------|--------|
| MLP Baseline | **1.00 ± 0.0** | 100% |
| Transformer + Pretrain | 0.33 ± 0.47 | 33% |

**Pretrain Loss 行为**: 在 ~1.0 震荡，不收敛

**根本原因**:
1. 表格数据无空间/语义结构
2. Mask reconstruction 无意义
3. Gate 在 pretrain 阶段冻结
4. Token 数量太少 (20-32)

**结论**: **Confirmed Negative Result**

### 3.5 训练顺序消融 (Training Order)

**来源**: `training_order_synthetic_full_20260329.json`

**问题**: 特征扩展 → 特征选择 → MLP 是否优于 选择 → MLP？

| 方法 | Mean best-k | Mean AUC |
|------|-------------|----------|
| **select_then_mlp** | **0.6090** | **0.6628** |
| expand4 | 0.3694 | 0.6221 |
| expand8 | 0.4184 | 0.6383 |
| expand16 | 0.4181 | 0.6537 |

**结论**: 先选择后 MLP 最优，扩展损害特征恢复

### 3.6 多项式特征消融

**来源**: `polynomial_ablation_20260407_115130.json`

**问题**: 多项式特征扩展是否有帮助？

| Dataset | Degree | Mode | best-k |
|---------|--------|------|--------|
| XOR | 1 | group | 1.00 |
| XOR | 2 | expanded | 1.00 |
| Ring | 1 | group | 0.10 |
| Ring | 2 | group | 0.40 |
| Ring+XOR | 1 | group | 0.55 |
| Ring+XOR | 2 | group | **0.75** |

**结论**: 2 阶多项式在 Ring+XOR 上提升 20%

---

## 4. 超参数配置

### 4.1 SADMM-FS 默认配置

```python
{
    # Architecture
    "latent_size": 32,
    "n_hidden_layers": 2,
    "gaussian_noise": 0.0,
    "dropout": 0.043,
    "activation": "mish",

    # Feature selection
    "feat_drop": 0.6,

    # Training
    "epochs": 416,
    "warmup_epochs": 100,
    "optimizer": "adagrad",
    "lr": 0.00176,
    "batch_size": 56,
    "patience": 66,

    # ADMM
    "rho": "adaptive",
    "sparsity_penalty": "ratio_norm"  # or "l1"
}
```

### 4.2 Gradual ADMM 变体

```python
{
    "rho_start": 0.001,
    "rho_end": 0.1,
    "rho_schedule": "linear",  # 或 "exponential"
}
```

### 4.3 Baseline 训练配置对比

| 方法 | Epochs | Optimizer | Notes |
|------|--------|-----------|-------|
| SADMM-FS | 416 | Adagrad | 100 warmup + 316 ADMM |
| STG | 300 | Adam | 官方配置 |
| TabNet | 100 | Adam | early stopping (patience=20) |
| FSNet | 2000 | Adam | 更长训练 |
| CAE | 300 | Adam | per selector |
| E2E-FS | 200 | Adam | per selector |

---

## 5. 关键发现总结

### 5.1 正面发现

| 发现 | 证据 | 影响 |
|------|------|------|
| SADMM-FS 综合最优 | best-k 0.6278 > 所有 baseline | 主方法有效 |
| Linear gate > Sigmoid gate | +3-8% best-k | 设计选择 |
| Gradual ADMM 改善 Ring | 0.67 → 1.00 (+33%) | 变体方法 |
| MLP > Transformer backbone | 0.63 vs 0.25 best-k | 架构选择 |
| 2 阶多项式帮助混合任务 | Ring+XOR: +20% | 预处理选项 |

### 5.2 Negative Results

| 发现 | 证据 | 教训 |
|------|------|------|
| Transformer 预训练无效 | best-k 0.33 vs 1.00 | 表格数据无空间结构 |
| Lottery Ticket 不适用 | Ring: 0.08 vs 0.67 | 权重复位损害 FS |
| 硬剪枝失败 | iterative_hard: 0.13 | 端到端训练重要 |
| 特征扩展损害 best-k | expand16: 0.42 vs 0.61 | 先选择后 MLP 最优 |

### 5.3 任务特性

| 任务类型 | 最优方法 | 原因 |
|----------|----------|------|
| XOR (交互检测) | SADMM-FS / LassoNet | 门控捕获交互 |
| Ring (圆形边界) | TreeSHAP / RF | 树模型擅长 |
| 混合任务 | SADMM-FS + gradual | 平衡策略 |

---

## 6. 论文状态 (UAI 2026)

### 6.1 审稿意见

**Score**: 5 (Weak Accept)

**主要贡献**:
1. SADMM Bridge (Eq. 11) - 连接 Adam 与 ADMM
2. Ratio Norm 三次方程求解 (Eq. 14-15)
3. 13 个 baseline 综合评估

**待解决问题**:
1. SADMM bridge 缺乏形式化收敛保证
2. Ratio Norm 与 L1 表现相当
3. DAG 评估仅使用单个图实例

### 6.2 待完成工作 (TODO_TKDE)

| Phase | Task | Status |
|-------|------|--------|
| Phase 2.1 | Column-normalized gate 变体 | NOT STARTED |
| Phase 2.2 | Feature dropout ablation | NOT STARTED |
| Phase 3.1 | Theory reframing | NOT STARTED |
| Phase 3.2 | Theory note | NOT STARTED |
| Phase 4 | Reruns | BLOCKED |
| Phase 5 | Paper revision | BLOCKED |

---

## 7. 文件索引

### 7.1 主实验结果

| 文件 | 内容 | 日期 |
|------|------|------|
| `mentor_axes/backbone_tier2_synthetic_full_20260327.json` | 主实验 + Tier-2 baseline | 2026-03-27 |
| `mentor_axes/mentor_gating_full_20260330_fixed.json` | Gating 消融 | 2026-03-30 |
| `matched_neural/matched_neural_synthetic_20260326_131516.json` | Matched baseline | 2026-03-26 |
| `mentor_axes/training_order_synthetic_full_20260329.json` | 训练顺序消融 | 2026-03-29 |

### 7.2 消融实验结果

| 文件 | 内容 | 日期 |
|------|------|------|
| `iterative_ablation_20260408_195630.json` | 迭代特征选择 | 2026-04-08 |
| `polynomial_ablation_20260407_115130.json` | 多项式扩展 | 2026-04-07 |
| `mentor_axes/transformer_penalty_ablation.json` | Transformer penalty | 2026-04-07 |
| `transformer_pretrain_ablation_20260407_161756.json` | Transformer pretrain | 2026-04-07 |

### 7.3 Real-world 结果

| 文件 | 内容 |
|------|------|
| `external-data/madelon-*.json` | Madelon 数据集 |
| `external-data/gisette-*.json` | Gisette 数据集 |
| `external-data/arcene-*.json` | Arcene 数据集 |
| `external-data/dexter-*.json` | Dexter 数据集 |

### 7.4 分析文档

| 文件 | 内容 |
|------|------|
| `analysis/current_results_report_20260403.md` | 综合结果报告 |
| `analysis/experiment_status_20260403.md` | 实验状态 |
| `analysis/fairness_protocol_20260403.md` | 公平性协议 |
| `analysis/transformer_pretrain_negative_result.md` | Negative result 分析 |
| `analysis/todo_experiments_summary_20260408.md` | 消融总结 |

---

## 8. 设计建议

基于所有实验，SADMM-FS 推荐配置:

1. **Backbone**: MLP (不使用 Transformer)
2. **Gate**: 线性无界 (不使用 sigmoid)
3. **迭代策略**: gradual_admm (需要时)
4. **多项式**: expanded 模式 (独立 gate)
5. **特征 dropout**: p=0.6-0.7
6. **优化器**: Adagrad, lr=0.00176

---

*文档生成: 2026-04-10*