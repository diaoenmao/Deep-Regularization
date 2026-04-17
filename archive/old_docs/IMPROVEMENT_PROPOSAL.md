# ADMM Input Group 改进建议

## 问题分析

### 当前性能对比 (N=1000, avg over m)

| 方法 | RING | XOR | RING+XOR | RING+XOR+SUM | AVG |
|------|------|-----|----------|--------------|-----|
| **ADMM Input Group (Ours)** | **13.0%** | 39.4% | 30.4% | **56.1%** | 34.7% |
| LassoNet | 34.8% | **81.8%** | 44.6% | 64.2% | 56.3% |
| CancelOut (sigmoid) | 34.1% | 60.6% | 37.5% | 55.8% | 47.0% |
| Random Forest | 100.0% | 54.5% | 88.8% | 85.3% | 82.2% |

### 关键发现

1. **RING+XOR+SUM 表现最好 (第 5 名)** - 说明方法在复杂多信号场景有效
2. **RING 最差 (13.0%, 垫底)** - 几何环形边界难以捕捉
3. **XOR 中等 (39.4%)** - 远低于 LassoNet (81.8%)

---

## 与 SOTA 方法的差异分析

### 1. LassoNet 的关键特点

```python
# LassoNet 配置
hidden_dims=(32, 32),      # 2 层，每层 32 单元
n_iters=(30, 30),          # 每层 30 次迭代
batch_size=64,
dropout=0,                 # 无 dropout
patience=10,               # 早停
lambda_start=3.2768,       # L1 正则起始值
```

**差异：**
| 特性 | LassoNet | ADMM Input Group |
|------|----------|-----------------|
| 架构 | 2 层 × 32 | 5 层 × 58 |
| 特征选择 | 跳跃层 L1 约束 | 输入层 gate + Ratio Norm |
| 优化 | 路径优化 (path) | ADMM 2 步优化 |
| 早停 | ✓ (patience=10) | ✗ (固定 500 epochs) |
| Dropout | 0 | 0.043 + feat_drop=0.6 |

**可能原因：**
- LassoNet 的**浅层架构** (2 层) 更适合小样本 (N=1000)
- **路径优化** 自动选择最佳 λ，我们的 C=0.05 是固定的
- **早停机制** 防止过拟合

---

### 2. CancelOut 的关键特点

```python
# CancelOut 配置
self.weights = Parameter(torch.zeros(input_size) + beta)  # beta=1
activation='sigmoid'  # 或 softmax
lambda1=0.2  # 方差正则
lambda2=0.1  # L1 正则
```

**CancelOut 的 weight_loss：**
```python
loss = -lambda1 * torch.var(w) + lambda2 * torch.sum(w)
```

**差异：**
| 特性 | CancelOut | ADMM Input Group |
|------|-----------|-----------------|
| 门控初始化 | β=1 (sigmoid 后≈0.73) | 1.0 (无偏置) |
| 正则化 | 方差最大化 + L1 | Ratio Norm (L1/L2) |
| 优化 | 端到端 Adam | ADMM 分步优化 |
| 门控激活 | sigmoid/softmax | sigmoid (可选) |

**关键洞察：**
- CancelOut 的**方差最大化** (`-torch.var(w)`) 强制 gate 值分化
- 我们的 Ratio Norm 是 scale-invariant，但可能不够强制稀疏
- CancelOut 初始化 β=1 让 gate 从中间值开始，更易学习

---

### 3. Benchmark 默认 NN 架构

```python
# Feature-Selection-Benchmark 的 Model
gaussian_noise=0.7466805127272365  # 高噪声！
dropout=0.04308691548552568
latent_size=58
n_hidden_layers=5
activation='mish'
optimizer='adagrad'  # 默认 Adagrad
learning_rate=0.00176
weight_decay=0.000485
epochs=416
patience=66  # 早停
```

**我们的配置：**
```python
# ADMM Input Group
gaussian_noise=0.0  # 禁用！
dropout=0.04308691548552568
latent_size=58
n_hidden_layers=5
activation='mish'
optimizer='adam'  # Adam
lr=0.005
warmup_epochs=120
epochs=500
```

**差异：**
1. **高斯噪声**: Benchmark 用 0.747，我们用 0.0
2. **优化器**: Benchmark 用 Adagrad，我们用 Adam
3. **早停**: Benchmark 有 patience=66，我们固定 500 epochs

---

## 改进建议

### 优先级 1: 添加早停机制 (高优先级)

**问题**: 固定 500 epochs 可能导致过拟合或欠拟合

**改进**:
```python
# 在 _train_input_group 中添加
best_val_loss = float('inf')
patience = 66
no_improve_count = 0

# 每 epoch 计算 validation loss
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = copy.deepcopy(model.state_dict())
    no_improve_count = 0
else:
    no_improve_count += 1
    if no_improve_count >= patience:
        break

# 恢复最佳状态
model.load_state_dict(best_state)
```

---

### 优先级 2: 添加高斯噪声 (高优先级)

**问题**: 禁用高斯噪声可能让模型在低维过拟合

**改进**:
```python
# GatedFeatureSelectionMLP.__init__
gaussian_noise: float = 0.7466805127272365,  # 改回 benchmark 默认值

# 或在 forward 中
if self.training and self.gaussian_noise > 0:
    x = x + self.gaussian_noise * torch.randn_like(x)
```

---

### 优先级 3: 改进 Ratio Norm 正则 (中优先级)

**问题**: 当前 Ratio Norm 可能不够强制 gate 分化

**改进方案 A - 添加方差正则**:
```python
# 在 ADMM 的 g-step 添加
gate_var_loss = -0.2 * torch.var(gate_param)  # CancelOut 风格
total_loss = data_loss + admm_penalty + gate_var_loss
```

**改进方案 B - 动态调整 C**:
```python
# 类似 LassoNet 的路径优化
C_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
best_C = 0.05
for C in C_values:
    # 训练并评估
    if val_auc > best_auc:
        best_C = C
```

---

### 优先级 4: 优化器选择 (中优先级)

**问题**: Adam 可能不如 Adagrad 稳定

**改进**:
```python
# 测试 Adagrad
opt_all = torch.optim.Adagrad(
    model.parameters(),
    lr=0.00176,  # benchmark 默认
    weight_decay=0.000485
)
```

---

### 优先级 5: 架构调整 (低优先级)

**减少层数测试**:
```python
# 测试 2 层架构 (类似 LassoNet)
n_hidden_layers=2
latent_size=32
```

---

## RING 数据集特别改进

RING 垫底 (13%) 的可能原因：

1. **环形边界需要非线性** - 5 层 MLP 可能过度参数化
2. **特征交互复杂** - 单一 gate 可能不够

**建议测试**:
```python
# 方案 A: 减少层数
model = GatedFeatureSelectionMLP(
    n_hidden_layers=2,  # 从 5→2
    latent_size=32,     # 从 58→32
)

# 方案 B: 添加特征交互
class GatedFeatureSelectionMLPv2(nn.Module):
    def __init__(self, ...):
        self.gate = Parameter(ones(n_features))
        self.pairwise_gate = Parameter(ones(n_features*(n_features-1)/2))

    def forward(self, x):
        g = sigmoid(self.gate)
        x = x * g
        # 添加二阶特征
        x2 = x[:, :1] * x[:, 1:2]  # 示例
        return self.layers(x)
```

---

## 实验计划

### Phase 1: 快速验证 (1-2 天)
1. 添加早停 (patience=66)
2. 恢复高斯噪声 (0.747)
3. 测试 Adagrad vs Adam

### Phase 2: 正则化调优 (3-5 天)
1. 测试 C 值路径：[0.01, 0.02, 0.05, 0.1, 0.2]
2. 添加方差正则 (lambda1=0.2)
3. 测试不同 feat_drop: [0.5, 0.6, 0.7]

### Phase 3: 架构探索 (5-7 天)
1. 减少层数 (2 层 vs 5 层)
2. 测试更小 latent_size (32 vs 58)
3. RING 特定：添加二阶特征交互

---

## 预期提升

| 改进 | 预计 RING 提升 | 预计 AVG 提升 |
|------|--------------|-------------|
| 早停 + 噪声 | +5-10% | +3-5% |
| 方差正则 | +3-5% | +2-3% |
| C 值调优 | +5-8% | +3-5% |
| 架构优化 | +10-15% | +5-8% |
| **总计** | **+23-38%** | **+13-21%** |

**目标**: RING 达到 35-50%, AVG 达到 45-55%

---

## 参考代码

### 早停实现
```python
# 在 _train_input_group 函数中
def _train_input_group(..., patience=66, use_val=True):
    ...
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # 训练
        ...

        # 验证
        if use_val and epoch % 10 == 0:
            val_loss = compute_val_loss(model, X_val, y_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stop at epoch {epoch}")
                    break

    # 恢复最佳状态
    if best_state:
        model.load_state_dict(best_state)
```

### 方差正则
```python
# 在 g-step 添加
gate_var = torch.var(gate_param)
gate_reg_loss = -0.2 * gate_var + 0.1 * torch.sum(torch.sigmoid(gate_param))
total_loss = data_loss + admm_penalty + gate_reg_loss
```
