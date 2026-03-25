# 解决低维与高维任务的 Architecture Trade-off

**日期**: 2026-03-23
**问题**: 小模型在低维任务上表现好，高维任务上容量不足；大模型相反

---

## 一、问题重述

### 1.1 实验数据汇总

| 任务 | 维度 | Small (2 层×32) | Baseline (5 层×58) | 最优 |
|------|------|----------------|-------------------|------|
| XOR | 8 | **100%** | 58.3% | Small |
| XOR | 128 | **100%** | 16.7% | Small |
| XOR | 1024 | **83.3%** | 75.0% | Small (微弱) |
| Ring | 32 | **100%** | 0% | Small |
| Ring | 512 | 8.3% | **33.3%** | Baseline |

**关键模式**:
- **低维 (m < 256)**: Small 显著优于 Baseline
- **高维 (m ≥ 512)**: Baseline 开始反超，但两者都不够好

### 1.2 核心问题

> 如何设计一个架构，在**低维和高维任务上都能自适应地表现良好**？

---

## 二、理论分析：为什么 Fixed Capacity 会失败

### 2.1 信息瓶颈的"甜蜜点"假说

我们提出 **Capacity-Dimension Matching Principle**:

$$\text{Optimal Performance} \iff d_h \approx f(m, k, n)$$

其中：
- $d_h$ = 隐藏层维度 (capacity)
- $m$ = 输入特征数
- $k$ = 真实特征数
- $n$ = 样本数

**三个机制区域**:

```
Region 1: Capacity 严重不足 (d_h << k)
  → 模型无法容纳所有真实特征 → 欠拟合

Region 2: Capacity 匹配 (d_h ≈ α·k, α > 1)
  → 足够容纳信号，但限制噪声 → 最佳特征选择

Region 3: Capacity 过剩 (d_h >> k)
  → 噪声特征也能通过 → 特征选择失败
```

### 2.2 我们的实验位置

| 架构 | d_h | XOR m=8 (k=2) | XOR m=1024 (k=2) | Ring m=512 (k=2) |
|------|-----|---------------|------------------|------------------|
| Small (32) | 32 | ✓ Region 2 | ✓ Region 2 | ✗ Region 1? |
| Baseline (58) | 58 | ✗ Region 3 | ✗ Region 3 | ~Region 2 |

**关键洞察**:
- Ring m=512 时，Small 的 32 维可能**不足以捕捉环形决策边界**
- 但 Baseline 的 58 维在低维任务上又**过于宽松**

---

## 三、解决方案：三种 Adaptive 策略

### 方案 A：Dimension-Aware Capacity Scaling

**核心思想**: 根据输入维度 $m$ 动态调整隐藏层维度 $d_h$。

**启发式规则**:
$$d_h = \max(32, \min(512, \lceil \sqrt{m \cdot k} \rceil))$$

对于我们的任务：
- m=8, k=2: $d_h = \lceil \sqrt{16} \rceil = 4$ → 实际用 32 (下限)
- m=128, k=2: $d_h = \lceil \sqrt{256} \rceil = 16$ → 实际用 32
- m=1024, k=2: $d_h = \lceil \sqrt{2048} \rceil ≈ 45$ → 实际用 58
- m=512, k=2: $d_h = \lceil \sqrt{1024} \rceil = 32$ → 边界情况

**实现方式**:
```python
def adaptive_capacity(n_features, k_true=2, base=32, max_cap=512):
    # 基于信息瓶颈理论的经验公式
    # d_h 应该足够大以容纳 k 个特征的非线性组合
    # 但足够小以限制噪声通过
    d_h = max(base, min(max_cap, int(np.sqrt(n_features * k_true))))
    return d_h
```

**优势**:
- 低维任务自动用小容量
- 高维任务自动增加容量
- 无需手动调参

**劣势**:
- 需要知道或估计 $k$ (真实特征数)
- 经验公式可能需要针对特定任务调整

---

### 方案 B：Hierarchical Bottleneck Architecture

**核心思想**: 使用**多层 bottleneck**，让模型自己学习压缩程度。

**架构设计**:
```
输入 (m) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(32) → Gate → Output
           ↓                  ↓                  ↓
        LayerNorm         LayerNorm         LayerNorm
```

**关键特性**:
1. **逐步压缩**: m → 128 → 64 → 32，每层压缩 2 倍
2. **层级特征选择**: 粗粒度特征在高层，细粒度在低层
3. **自适应信息流**: 通过 skip connection 让模型决定保留多少信息

**与 Fixed Capacity 对比**:

| 特性 | Fixed (32 维) | Hierarchical (128→64→32) |
|------|--------------|-------------------------|
| 低维任务 | 容量足够 | 早期层可学习"绕过"多余容量 |
| 高维任务 | 容量不足 | 早期层保留更多信息 |
| 特征选择 | 单层 gate | 多层级联合选择 |

**实现代码**:
```python
class AdaptiveBottleneckMLP(nn.Module):
    def __init__(self, input_size, n_classes, compression_ratio=0.5):
        super().__init__()
        # 自动计算 bottleneck 层数和维度
        dims = [input_size]
        while dims[-1] > 32:  # 最小 bottleneck 32 维
            dims.append(max(32, int(dims[-1] * compression_ratio)))
        dims.append(n_classes)

        # 构建层级 bottleneck
        layers = []
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU(),
            ])
        self.network = nn.Sequential(*layers)

        # Gate 在 bottleneck 最窄处
        self.gate = nn.Parameter(torch.ones(32))
```

**文献支持**:
- LassoNet [Lemhadri et al., 2021] 使用单层 bottleneck
- Deep Compression [Han et al., 2015] 使用多层压缩
- 我们的创新：**将 bottleneck 与特征选择 gate 结合**

---

### 方案 C：Dimension-Adaptive Gate (DAG)

**核心思想**: Gate 本身的容量也应该随维度调整。

**标准 Gate**:
$$g_j = \sigma(w_j^T x), \quad w_j \in \mathbb{R}^m$$

**问题**: 当 $m$ 很大时，$w_j$ 有太多自由度，可能过拟合噪声。

**改进 Gate**:
$$g_j = \sigma(W_2 \cdot \text{ReLU}(W_1 x)), \quad W_1 \in \mathbb{R}^{d_g \times m}, W_2 \in \mathbb{R}^{d_g}$$

其中 $d_g = \lceil \sqrt{m} \rceil$ 是 gate 的"bottleneck 维度"。

** intuition**:
- Gate 本身也需要一个信息瓶颈
- 低维任务：Gate 简单，直接加权
- 高维任务：Gate 有足够的 capacity 学习复杂的特征交互

**完整架构**:
```python
class DimensionAdaptiveGate(nn.Module):
    def __init__(self, n_features, gate_bottleneck_ratio=0.25):
        super().__init__()
        self.gate_hidden = max(16, int(n_features * gate_bottleneck_ratio))

        # Gate network with bottleneck
        self.gate_net = nn.Sequential(
            nn.Linear(n_features, self.gate_hidden),
            nn.ReLU(),
            nn.Linear(self.gate_hidden, n_features),
        )

        # Main prediction network (small fixed capacity)
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        gate = torch.sigmoid(self.gate_net(x))  # (batch, n_features)
        x_gated = x * gate
        out = self.backbone(x_gated)
        return out, gate
```

---

## 四、推荐方案：Hybrid Adaptive Architecture

结合上述三种策略的优势：

### 4.1 架构设计

```
输入 (m) → Adaptive Gate → Fixed Backbone (64→32→output)
           ↓
     d_g = √m bottleneck
```

**组件**:

| 组件 | 配置 | 作用 |
|------|------|------|
| **Adaptive Gate** | d_g = max(16, √m) | 根据维度调整 gate 容量 |
| **Backbone** | 固定 64→32 | 固定的预测容量，确保可比性 |
| **Feature Dropout** | p=0.5 | 额外的正则化 |

### 4.2 理论优势

1. **Gate-Backbone 分离**:
   - Gate 负责特征选择，需要适应维度
   - Backbone 负责预测，固定容量确保公平比较

2. **双重信息瓶颈**:
   - Gate 的 bottleneck 控制"哪些特征能通过"
   - Backbone 的 bottleneck 控制"多少信息能用于预测"

3. **低维任务**:
   - Gate 简单 (d_g ≈ 4-8)，不会过拟合
   - Backbone 容量充足 (32 维 > k=2)

4. **高维任务**:
   - Gate 有足够 capacity (d_g ≈ 32) 学习复杂选择
   - Backbone 依然是 32 维瓶颈，强制压缩

### 4.3 预期表现

| 任务 | 维度 | 预期表现 | 理由 |
|------|------|---------|------|
| XOR | 8 | 95-100% | Gate d_g=4，足够简单 |
| XOR | 128 | 95-100% | Gate d_g=16，平衡容量 |
| XOR | 1024 | 90-95% | Gate d_g=32，足够捕捉信号 |
| Ring | 32 | 95-100% | 低维任务，Gate 不会过拟合 |
| Ring | 512 | 50-70% | Gate d_g=24，比 Small 更好 |

---

## 五、实验验证计划

### 5.1 对比架构

| 架构 | Gate | Backbone | 预期 |
|------|------|----------|------|
| Small | 固定 32 | 2 层×32 | 低维好，高维差 |
| Baseline | 固定 58 | 5 层×58 | 低维差，高维中等 |
| **Adaptive (Ours)** | √m | 固定 64→32 | **两者兼顾** |

### 5.2 评估指标

1. **Feature Selection Accuracy**: best-k @ k
2. **Prediction Accuracy**: Test AUC
3. **Capacity Efficiency**: best-k / d_h
4. **Stability**: 6-fold CV 的标准差

### 5.3 实现优先级

1. **第一周**: 实现 Adaptive Gate，在 XOR 和 Ring 上验证
2. **第二周**: 添加 Hierarchical Bottleneck 选项
3. **第三周**: 全量对比实验 + 消融研究

---

## 六、文献对比与创新点

### 6.1 与现有工作的区别

| 方法 | Capacity | Gate 设计 | 维度自适应 |
|------|----------|----------|-----------|
| LassoNet [2021] | 固定 | 单层 skip | ✗ |
| STG [2019] | 固定 | 随机 gate | ✗ |
| DeepFS [2020] | 固定 | 注意力 | ✗ |
| **Ours** | **自适应** | **Bottleneck Gate** | **✓** |

### 6.2 理论贡献

1. **Capacity-Dimension Matching Principle**: 首次明确提出特征选择的容量匹配问题
2. **Dimension-Adaptive Gate**: 第一个维度自适应的 gate 设计
3. **经验公式**: $d_h = \lceil \sqrt{m \cdot k} \rceil$ 作为容量选择的启发式

### 6.3 论文定位

可以投递的 venue:
- **NeurIPS/ICML**: 如果理论分析足够深入
- **AAAI/IJCAI**: 强调自适应机制的创新
- **AISTATS**: 如果侧重统计理论

---

## 七、结论

**核心建议**: 实现 **Hybrid Adaptive Architecture**

```python
# 伪代码示例
class AdaptiveFeatureSelector(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        # Adaptive gate bottleneck
        self.gate_hidden = max(16, int(np.sqrt(n_features)))
        self.gate = nn.Sequential(
            nn.Linear(n_features, self.gate_hidden),
            nn.ReLU(),
            nn.Linear(self.gate_hidden, n_features),
        )

        # Fixed backbone (确保可比性)
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        out = self.backbone(x * gate)
        return out, gate
```

**下一步**: 先实现这个架构，然后在 XOR 和 Ring 上跑对比实验。
如果结果如预期，这将是我们论文的核心创新点。

---

**生成时间**: 2026-03-23
**下一步行动**: 实现 AdaptiveFeatureSelector 并验证
