# Custom ADMM Results 索引

**最后更新**: 2026-04-10

---

## 目录结构

```
results/
├── mentor_axes/           # 主要实验结果
├── matched_neural/        # Matched baseline 对比
├── external-data/         # Real-world 数据集结果
├── ablations/             # Dropout 消融
├── tuning/                # 超参调优
├── validation/            # 验证实验
├── iterative_ablation_*.json   # 迭代特征选择消融
├── polynomial_ablation_*.json  # 多项式扩展消融
├── transformer_pretrain_*.json # Transformer 预训练消融
└── *.txt                  # 文本日志
```

---

## 主实验结果

### mentor_axes/ (核心实验)

| 文件 | 内容 | 日期 |
|------|------|------|
| `backbone_tier2_synthetic_full_20260327.json` | **主实验**: SADMM-FS vs Tier-2 baselines | 2026-03-27 |
| `mentor_gating_full_20260330_fixed.json` | Gating 消融: Linear vs Sigmoid | 2026-03-30 |
| `training_order_synthetic_full_20260329.json` | 训练顺序消融 | 2026-03-29 |
| `tier2_rerun_fixed_20260402.json` | Tier-2 重跑验证 | 2026-04-02 |
| `transformer_penalty_ablation.json` | Transformer penalty 消融 | 2026-04-07 |

### matched_neural/ (Matched Baseline)

| 文件 | 内容 |
|------|------|
| `matched_neural_synthetic_20260326_131516.json` | SADMM-FS vs STG/CancelOut/DeepPINK |

### external-data/ (Real-world)

| 数据集 | 方法 |
|--------|------|
| madelon | admm_input_group, stg |
| gisette | admm_input_group, stg |
| arcene | admm_input_group, stg |
| dexter | admm_input_group, stg |
| fashion | admm_input_group, stg |
| isolet | admm_input_group, stg |
| har | admm_input_group, stg |
| coil20 | admm_input_group, stg |
| mice | admm_input_group, stg |

---

## 消融实验

### 迭代特征选择 (Iterative FS)

| 文件 | 内容 |
|------|------|
| `iterative_ablation_20260408_195630.json` | **最新**: single_pass/iterative_hard/lottery_ticket/gradual_admm |

**结果**:
- gradual_admm 在 Ring 上 best-k=1.00 (vs single_pass 0.67)
- lottery_ticket 完全失败 (Ring best-k=0.08)

### 多项式扩展

| 文件 | 内容 |
|------|------|
| `polynomial_ablation_20260407_115130.json` | **最新**: degree 1-2, group/expanded mode |

**结果**:
- Ring+XOR: degree=2 提升 20%
- expanded mode > group mode

### Transformer Pretrain

| 文件 | 内容 |
|------|------|
| `transformer_pretrain_ablation_20260407_161756.json` | **最新**: MAE pretrain 无效 |

**结果**: MLP baseline best-k=1.00, Transformer+pretrain=0.33

---

## 快速查找

### 按实验类型

| 类型 | 文件位置 |
|------|----------|
| 主方法对比 | `mentor_axes/backbone_tier2_synthetic_full_*.json` |
| Gating 消融 | `mentor_axes/mentor_gating_full_*.json` |
| Baseline 对比 | `matched_neural/matched_neural_synthetic_*.json` |
| 迭代消融 | `iterative_ablation_*.json` |
| 多项式消融 | `polynomial_ablation_*.json` |
| Transformer 消融 | `transformer_pretrain_ablation_*.json` |
| Real-world | `external-data/*.json` |

### 按日期

| 日期 | 主要实验 |
|------|----------|
| 2026-03-26 | Matched neural baseline |
| 2026-03-27 | Backbone/Tier-2 主实验 |
| 2026-03-29 | Training order 消融 |
| 2026-03-30 | Gating 消融 |
| 2026-04-02 | Tier-2 重跑 |
| 2026-04-07 | Polynomial/Transformer 消融 |
| 2026-04-08 | Iterative 消融 (最新) |

---

## 关键数值

### SADMM-FS 主实验

| Metric | Value |
|--------|-------|
| Mean best-k | **0.6278** |
| Mean AUC | **0.6605** |
| XOR best-k | 1.000 |
| Ring best-k | 0.500 (gradual_admm: 1.000) |

### Baseline 对比

| Method | Mean best-k |
|--------|-------------|
| SADMM-FS | **0.6278** |
| TabNet | 0.2851 |
| CAE | 0.1861 |
| FSNet | 0.1667 |

---

*索引生成: 2026-04-10*