# 实验计划：Iterative & Expansion 特征选择方法探索

**创建日期**: 2026-04-10
**状态**: 计划阶段

---

## 1. 背景与动机

### 1.1 当前结果

**Iterative Feature Selection**:
| 方法 | XOR | Ring | Ring+XOR |
|------|-----|------|----------|
| single_pass | 1.00 | 0.67 | 0.67 |
| iterative_hard | 0.67 | 0.58 | 0.13 |
| lottery_ticket | 0.50 | **0.08** | 0.13 |
| gradual_admm | **1.00** | **1.00** | **0.67** |

**Polynomial Expansion**:
| Dataset | Degree=1 | Degree=2 (group) | Degree=2 (expanded) |
|---------|----------|------------------|---------------------|
| XOR | 1.00 | 0.67 | **1.00** |
| Ring | 0.10 | 0.40 | - |
| Ring+XOR | 0.55 | **0.75** | - |

### 1.2 核心问题

1. **Iterative**: 为什么 lottery_ticket 完全失败？soft pruning 能否改善？
2. **Expansion**: 如何更好地选择多项式特征？
3. **Combined**: 迭代 + 扩展能否协同？

---

## 2. 实验目标

### 主目标

1. 验证 **soft pruning** 是否优于 hard pruning
2. 验证 **hierarchical selection** 是否优于 group/expanded
3. 验证 **knowledge distillation** 是否保留更多信息
4. 探索 **group lasso** 对多项式特征的效果

### 次要目标

5. 测试 **curriculum learning** 是否改善 Ring 任务
6. 测试 **Fourier features** 对 Ring 任务的效果

---

## 3. 实验设计

### 3.1 Experiment A: Gradual ADMM + Pruning + Re-weighting

**问题**: Hard vs Soft pruning 的公平比较？Re-weighting 是否改善性能？

**核心问题**: 当前比较不公平
- Hard pruning: 输入维度减少 (m→k)，模型更小
- Soft pruning: 输入维度不变 (m)，仍处理弱特征噪声

**解决方案**: 加入 **Re-weighting** 使比较更公平

**Re-weighting 逻辑**:
```python
# 剪枝后 gate 能量下降，需要恢复
Before prune: sum(|gate|) = S_total
After prune:  sum(|gate_alive|) = S_current < S_total

# Re-weight: 保持存活特征的能量
scale = S_target / S_current
g_alive *= scale
```

**方法矩阵** (公平比较):

| 方法 | Prune | Re-weight | 输入维度 | 最终状态 |
|------|-------|-----------|----------|----------|
| `gradual_none` | No | No | m | m (gate→0) |
| `gradual_soft` | Soft mask | Yes | m | m (gate≈0 or ≈scale) |
| `gradual_soft_no_rw` | Soft mask | No | m | m |
| `gradual_hard` | Delete | Yes | m→k | k |
| `gradual_hard_no_rw` | Delete | No | m→k | k |

**实现**:
```python
def gradual_admm_with_pruning(
    model, X_train, y_train, n_classes,
    C_schedule=[0.001, 0.01, 0.05, 0.1],
    prune_threshold=0.01,
    prune_mode="soft",  # "soft", "hard", "none"
    reweight=True,
):
    target_gate_sum = model.gate.abs().sum().item()
    
    for phase, C in enumerate(C_schedule):
        # Phase 1: Train with current C
        train_admm(model, X_train, y_train, C=C)
        
        # Phase 2: Identify weak features
        gate = model.gate.detach()
        alive_mask = (gate.abs() >= prune_threshold)
        
        # Phase 3: Prune + Re-weight
        if prune_mode == "soft":
            model.gate.data = gate * alive_mask.float()
            if reweight:
                model.gate.data = reweight_gates(
                    model.gate.data, alive_mask.float(), target_gate_sum
                )
        elif prune_mode == "hard":
            alive_indices = alive_mask.nonzero().squeeze()
            model.gate = Parameter(model.gate[alive_indices])
            if reweight:
                model.gate.data *= (target_gate_sum / model.gate.abs().sum())
            X_train = X_train[:, alive_indices]
```

**数据集**: XOR, Ring, Ring+XOR (m=128, m=256)

**评估**: 6-fold CV, best-k, AUC, gate_sum, alive_count

**预期结果**:

| 方法 | XOR best-k | Ring best-k | Ring+XOR | gate_sum |
|------|------------|-------------|----------|----------|
| gradual_none | 1.00 | 1.00 | 0.67 | →0 |
| gradual_soft | 1.00 | 1.00 | 0.67 | stable |
| gradual_soft_no_rw | 1.00 | 0.90 | 0.60 | ↓ |
| gradual_hard | 1.00 | 0.85 | 0.55 | stable |
| gradual_hard_no_rw | 0.90 | 0.70 | 0.45 | ↓ |

**核心假设**: Re-weighting 改善所有方法，尤其是 hard pruning

---

### 3.2 Experiment B: Hierarchical Selection

**问题**: 两层 gate (原始 → 扩展) 是否优于单层？

**方法**:

| 方法 | 描述 |
|------|------|
| `group` | 一个 gate 控制原始特征的所有扩展 |
| `expanded` | 每个扩展特征独立 gate |
| `hierarchical` | 两层 gate: g_orig * g_exp |

**实现**:
```python
# Hierarchical: 两层选择
class HierarchicalGateModel:
    gate_original = Parameter(m)      # 原始特征 gate
    gate_expanded = Parameter(m_exp)  # 扩展特征 gate
    
    def forward(x_exp):
        g_effective = g_original[mapping] * g_expanded
        return x_exp * g_effective
```

**数据集**: XOR, Ring, Ring+XOR, degree=2

**评估**: 6-fold CV, best-k (原始特征空间)

**预期**: hierarchical ≥ expanded > group

---

### 3.3 Experiment C: Knowledge Distillation

**问题**: 蒸馏是否帮助保留被剪枝特征的信息？

**方法**:

| 方法 | 描述 |
|------|------|
| `no_distill` | 直接剪枝 (baseline) |
| `distill` | Teacher-student 蒸馏 |
| `self_distill` | 自蒸馏 (当前模型作为 teacher) |

**实现**:
```python
# Knowledge distillation loss
loss = alpha * task_loss + (1-alpha) * KL(student || teacher)
```

**数据集**: XOR, Ring (m=128)

**评估**: best-k, accuracy retention

**预期**: distill > no_distill

---

### 3.4 Experiment D: Group Lasso on Polynomial Features

**问题**: Group lasso 是否改善多项式特征选择？

**方法**:

| 方法 | 描述 |
|------|------|
| `l1` | 标准 L1 惩罚 |
| `ratio_norm` | Ratio Norm 惩罚 |
| `group_lasso` | 对扩展组施加 group penalty |
| `sparse_group` | Group lasso + L1 组内稀疏 |

**实现**:
```python
# Group lasso: 同一原始特征的扩展组
for orig_idx, exp_indices in expansion_groups.items():
    penalty += lambda_g * ||gate[exp_indices]||_2

# Sparse group lasso
penalty = lambda_g * group_penalty + lambda_1 * l1_penalty
```

**数据集**: Ring+XOR (degree=2)

**评估**: best-k, sparsity

**预期**: group_lasso ≥ ratio_norm > l1

---

### 3.5 Experiment E: Fourier Features (探索性)

**问题**: Fourier 特征是否帮助 Ring 任务？

**方法**:

| 方法 | 描述 |
|------|------|
| `polynomial_d2` | 2 阶多项式 |
| `fourier_10bands` | 10 频带 sin/cos |
| `fourier_20bands` | 20 频带 sin/cos |
| `combined` | Polynomial + Fourier |

**实现**:
```python
# Fourier features
for b in bands:
    features.append(sin(2π * b * x))
    features.append(cos(2π * b * x))
```

**数据集**: Ring (m=32, m=128)

**评估**: best-k, AUC

**预期**: fourier 对 Ring 可能有帮助 (圆形边界)

---

## 4. 实验矩阵

### 优先级排序

| 优先级 | 实验 | 工作量 | 预期收益 |
|--------|------|--------|----------|
| P0 | A: Soft vs Hard Pruning | 低 | 高 |
| P0 | B: Hierarchical Selection | 低 | 中 |
| P1 | C: Knowledge Distillation | 中 | 中 |
| P1 | D: Group Lasso | 中 | 中 |
| P2 | E: Fourier Features | 低 | 未知 |

### 依赖关系

```
A (soft pruning) ──┐
                   ├──> F: Combined (soft + hierarchical + group lasso)
B (hierarchical) ──┤
                   │
D (group lasso) ───┘

C (distillation) ──> G: Distillation + Soft Pruning

E (fourier) ──> 独立探索
```

---

## 5. 实现计划

### Phase 1: 代码实现 (1-2天)

```
custom_admm/src/
├── iterative_soft_pruning.py    # Experiment A
├── hierarchical_selection.py    # Experiment B (已有)
├── distillation_pruning.py      # Experiment C (已有)
├── group_lasso_expansion.py     # Experiment D
└── fourier_features.py          # Experiment E (已有)
```

### Phase 2: 实验运行 (2-3天)

| 实验 | 运行脚本 | 预计时间 |
|------|----------|----------|
| A | `run_experiment_A_soft_pruning.py` | 4h |
| B | `run_experiment_B_hierarchical.py` | 4h |
| C | `run_experiment_C_distillation.py` | 6h |
| D | `run_experiment_D_group_lasso.py` | 4h |
| E | `run_experiment_E_fourier.py` | 2h |

### Phase 3: 分析与报告 (1天)

- 汇总结果到 `analysis/experiment_iterative_expansion_results.md`
- 更新 `EXPERIMENTS_SUMMARY.md`

---

## 6. 成功标准

### 定量标准

| 实验 | 成功标准 |
|------|----------|
| A | soft_mask best-k > hard_prune by 10%+ |
| B | hierarchical best-k ≥ expanded |
| C | distill accuracy retention > 90% |
| D | group_lasso best-k > ratio_norm on Ring+XOR |
| E | fourier Ring best-k > polynomial |

### 定性标准

- 方法可复现
- 代码有文档
- 结果有统计分析

---

## 7. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Soft pruning 不稳定 | 使用更细粒度的阈值调度 |
| Hierarchical 过拟合 | 增加正则化 |
| Distillation 效果差 | 调整 alpha 和 temperature |
| Group lasso 计算慢 | 使用 proximal 算子优化 |
| Fourier 对 Ring 无效 | 尝试不同频带数量 |

---

## 8. 后续方向

如果 Phase 1-3 成功，可继续：

### Phase 4: 组合方法

1. **Soft + Hierarchical + Group Lasso** (A+B+D)
2. **Distillation + Soft Pruning** (C+A)
3. **Curriculum Learning**: 先线性，再加多项式，最后 Fourier

### Phase 5: 论文整合

- 选择最优方法作为 SADMM-FS 的增强版本
- 更新论文实验部分
- 添加消融分析

---

## 9. 文件结构

```
custom_admm/
├── src/
│   ├── iterative_run.py              # 已有 iterative 方法
│   ├── polynomial_expansion.py       # 已有 expansion 方法
│   ├── iterative_soft_pruning.py     # NEW: Experiment A
│   ├── group_lasso_expansion.py      # NEW: Experiment D
│   └── ...
├── run_experiment_A_soft_pruning.py
├── run_experiment_B_hierarchical.py
├── run_experiment_C_distillation.py
├── run_experiment_D_group_lasso.py
└── run_experiment_E_fourier.py

results/
├── experiment_A_soft_pruning_*.json
├── experiment_B_hierarchical_*.json
├── experiment_C_distillation_*.json
├── experiment_D_group_lasso_*.json
└── experiment_E_fourier_*.json
```

---

## 10. 立即行动项

### 第一步：实现 Experiment A (Soft Pruning)

```python
# custom_admm/src/iterative_soft_pruning.py

def soft_iterative_pruning(
    model,
    X_train, y_train, X_test, y_test,
    threshold_schedule: list[float],
    epochs_per_threshold: int = 50,
):
    """Soft pruning: mask features instead of removing them."""
    results = []
    
    for threshold in threshold_schedule:
        # Train
        train_model(model, X_train, y_train, epochs=epochs_per_threshold)
        
        # Soft mask (don't remove)
        with torch.no_grad():
            mask = (model.gate.abs() > threshold).float()
            model.gate.data = model.gate.data * mask
        
        # Evaluate
        scores = model.gate.abs().cpu().numpy()
        best_k = compute_best_k(scores, ground_truth)
        results.append({"threshold": threshold, "best_k": best_k})
    
    return results
```

---

*计划创建: 2026-04-10*
*预计完成: 2026-04-15*