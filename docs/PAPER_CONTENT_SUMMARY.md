# GRN-FS 技术报告

最后更新：2026-05-03

---

## 1. 一句话概述

GRN-FS（Gated Ratio-Norm Regularization for Feature Selection）是一种基于神经网络门控的特征选择训练流程。当前 paper-facing 版本使用 input-dimension weight-normalized gate：每个输入特征对应一个 normalized first-layer direction 和一个可稀疏化的 scalar gate。ADMM 把预测训练和稀疏支撑学习解耦，多阶段训练在 phase boundary 用 ADMM 辅助变量 `z` 的 support 写回 gate，从而把连续 gate 转成稳定的 sparse feature support。Degree-2 expansion 只是 synthetic nonlinear benchmark 的 optional interaction dictionary，不是方法本体。

---

## 2. 方法背景

特征选择的目标是从输入特征中选出少量有用特征，同时保持预测性能。GRN-FS 的基本做法是给每个输入特征分配一个可学习 gate：

```text
x_gated = x * g
```

如果某个 `g_j` 接近 0，第 `j` 个特征对模型几乎不起作用；如果 `g_j` 较大，该特征更重要。

方法的难点在于：神经网络训练自然会得到连续 gate，而特征选择最终需要一个明确的稀疏集合。因此 GRN-FS 结合了：

- 神经网络预测损失；
- ADMM 稀疏优化；
- 多阶段逐步增强稀疏压力；
- phase boundary 上的 ADMM-guided `z` support masking；
- 可选的 feature map，例如 synthetic nonlinear benchmark 上的 degree-2 expansion。

---

## 3. 核心方法

### 3.1 Gate 与 weight normalization

基础 gate 直接乘在输入上：

```text
x_gated = x * g
```

当前主方法使用 weight-normalized gate。它的核心不是“普通输入 mask 再让第一层权重自由吸收尺度”，而是把第一层每个输入列写成 **方向** 和 **可稀疏化尺度**：

```text
W_1[:, j] = g_j * Wbar_1[:, j]
Wbar_1[:, j] = v_j / ||v_j||_2
||Wbar_1[:, j]||_2 = 1
```

其中：

- `g_j` 是第 `j` 个特征的可学习 gate；
- `v_j` 是第一层神经网络中连接第 `j` 个输入特征的 raw direction parameter；
- `Wbar_1[:, j]` 是 normalized weight column，只保留方向信息；
- `g_j` 直接承担 weight normalization 里的 scalar length / feature scale 角色，因此也是 ADMM 要稀疏化的变量。

因此，概念上的第一层计算可以写成：

```text
h = Wbar_1 (x * g) + b_1
```

这个形式把两件事分开：

| 部分 | 含义 |
|---|---|
| `Wbar_1[:, j]` | normalized weight direction，表示该特征如何连接到 hidden units |
| `g_j` | learnable scalar gate，同时是该 normalized direction 的功能尺度 |
| `|g_j|` | paper-facing raw-feature ranking 的默认分数 |

需要注意，输入 gate 和第一层权重列 gate 在数学上等价：

```text
W_1 diag(g) x = W_1 (x * g)
```

但当前的 weight-normalized gate 不只是等价改写，因为第一层列方向被单位化，feature strength 不能再被 raw column norm 隐式吸收，只能通过显式 scalar gate `g_j` 表达。第三方读者可以把它理解为：**先把第一层权重列 normalized，再把 `g_j` 作为该 raw feature 的可训练尺度和选择变量。**

代码层面对应 `GatedWeightNormMLP.forward()`：先计算 `x_scaled = x * g`，再用 `first_normalized_weight` 完成第一层线性映射。`first_weight_norm` 可以作为诊断量读取，但当前 weight-norm gate 版本不会把 raw column norm 乘回 forward，也不会把它作为主 readout。

### 3.2 ADMM 稀疏优化

原始目标可以写成：

```text
min_{theta, g} L(theta, g) + C R(g)
```

其中：

- `L(theta, g)` 是预测损失；
- `R(g)` 是稀疏惩罚；
- `C` 控制稀疏强度。

在当前 weight-normalized gate 版本里，`g_j` 本身就是 normalized input-column 的 scalar scale，因此主稀疏变量直接是 `g`。早期 input-gate / adaptive-threshold 诊断会使用第一层 column norm 构造 `lambda_j = C / s_j`；但 `GatedWeightNormMLP` 的当前实现把列方向单位化，主 readout 和主 gate penalty 都应围绕 `g_j` 本身解释。

ADMM 引入辅助变量 `z`：

```text
min_{theta, g, z} L(theta, g) + C R(z)
subject to g = z
```

每个 epoch 主要包含三步：

| 步骤 | 更新对象 | 作用 |
|---|---|---|
| prediction/gate step | `theta`, `g` | 学习预测模型和 gate |
| proximal step | `z` | 对稀疏惩罚做近端更新，产生稀疏辅助变量 |
| dual update | `u` | 推动 `g` 与 `z` 一致 |

这里 `z` 不只是训练时的辅助变量。proximal step 会直接在 `z` 上产生严格零，因此 `z` 的 support 是 ADMM 对“哪些特征应该保留”的离散判断。当前推荐方法在 phase boundary 使用这个 support 来更新 `g`，而不是只对 `g` 本身做固定阈值。

### 3.3 Phase-boundary `z` masking

ADMM 的 proximal step 会让辅助变量 `z` 出现严格零，但模型真正使用的是 `g`。在有限训练轮数下，`g` 往往只是接近 0，而不是严格等于 0：

```text
z_j = 0
|g_j| may still be around 1e-3 to 1e-2
```

早期版本在每个 phase 结束后直接对 `g` 做阈值：

```text
if |g_j| < eta:
    g_j <- 0
```

这个规则可以把 approximate sparse gate 转成 strict sparse gate，但它没有直接使用 ADMM 已经算出的 `z`。当前 paper-facing 方法把 ADMM 的 `z` support 作为 phase-boundary 的离散选择信号：

```text
if z_j == 0:
    g_j <- 0
```

也就是 `mask_by_z` / ADMM-support masking：`z` 决定哪些坐标应严格置零，`g` 保留 prediction/gate step 学到的连续幅度。这个设计比直接 `g <- z` 更合适，因为 `g <- z` 会丢掉 prediction/gate step 学到的 gate magnitude；support masking 只使用 `z` 的严格零结构。

需要区分两个实验版本：

| 版本 | 含义 | 当前定位 |
|---|---|---|
| `mask_by_z` | 每个 phase boundary 用 `z_j != 0` mask `g` | paper-facing algorithm / figure 的简洁主流程 |
| `mask_by_z_final` | 中间 phase 用 threshold，最后 phase 才用 `z` mask | synthetic ablation 中表现很强的变体，可作为诊断结果报告 |

`z` masking 和 hard pruning 的区别：

| 操作 | 含义 | 是否可逆 |
|---|---|---|
| `mask_by_z` | phase boundary 把 `z_j = 0` 的 gate 置零，但保留特征和参数 | paper-facing support conversion |
| hard pruning | 直接删除特征或模型结构 | 不可逆 |

实验显示，phase-boundary pruning 的核心价值不是单纯提升 AUC，而是把 ADMM 的离散支撑信息转移到模型实际使用的 `g` 上，让 feature readout 更稳定。`C` 在这个机制里尤其关键：`C` 太小会让 `z` mask 不够稀疏，`C` 太大则会把 support 压得过窄。

### 3.4 Feature map / polynomial expansion

GRN-FS 的核心训练机制不要求 degree-2 expansion。它可以直接作用在 raw features 上，也可以作用在某个显式 feature map 上：

```text
x_model = Phi(x)
x_gated = x_model * g
```

在 synthetic nonlinear benchmark 上，为了给 predictor 足够的非线性表达能力，我们使用 degree-2 polynomial expansion：

```text
Phi_2(x) = [x_1, ..., x_m, x_1^2, x_1 x_2, ..., x_m^2]
```

当原始特征数 `m = 128` 时，扩展后特征数为：

```text
128 + 128 * 129 / 2 = 8384
```

这个扩展对 Ring 等需要二次边界的预测 AUC 很重要，但它不是 GRN-FS 的通用必要组件。raw-feature ablation 显示，不用 expansion 时 feature recovery 仍然很强；下降主要发生在 nonlinear predictive AUC。只保留平方项的 diagonal-only expansion 维度较低，但缺少 cross terms，无法稳定覆盖 XOR 类交互任务，因此暂时不作为主线。

### 3.5 Multi-phase training

主方法不是一次性训练到结束，而是使用多个 phase。每个 phase 包含：

1. warmup：不加 ADMM 稀疏压力，让模型先适应该 phase 的状态；
2. ADMM training：使用当前强度 `C` 训练；
3. phase-boundary support masking：用 `z_j = 0` 的位置把对应 `g_j` 显式置零，同时保留其余 gate magnitude 和模型权重。

稀疏强度 `C` 随 phase 逐步增加。实验显示，简单地在 single-run 中连续 anneal `C(t)` 不能替代这种 multi-phase 机制。

### 3.6 Paper-facing 伪代码

论文里的 Algorithm 和 Figure 1 应该使用同一个流程。当前推荐伪代码如下：

```text
Algorithm: GRN-FS with Ratio Norm

Input:
  data (X, y)
  phase schedule {(T_w[p], T_a[p], C[p])}_{p=1}^P
  initial ADMM penalty rho_0
  readout rule

Initialize:
  predictor parameters theta
  gate g <- 1

for phase p = 1 ... P:
  z <- g
  u <- 0
  rho <- rho_0
  reset optimizer state

  # Warm-up stage
  for epoch = 1 ... T_w[p]:
    update theta, g with Adam on prediction loss L(theta, g)

  # ADMM stage
  for epoch = 1 ... T_a[p]:
    lambda <- C[p]  # current weight-normalized gate version

    g-step:
      update theta, g with Adam/AdaGrad on
        L(theta, g) + (rho / 2) ||g - z + u||_2^2

    z-step:
      q <- g + u
      z_tilde_j <- soft_threshold(q_j, lambda / rho)
      tau <- positive root of tau^3 - tau - D = 0
      z <- tau * z_tilde

    dual/rho update:
      u <- u + g - z
      every 5 epochs, adapt rho by residual balancing

  # Phase boundary
  g_j <- g_j * 1[z_j != 0]

Return:
  original-feature ranking by |g_j| for original inputs,
  or parent-sharing / aggregate original-space readout for polynomial inputs.
```

Figure 1 应该画成这个流程，而不是只画单次 ADMM：输入映射 `Phi(x)`，phase 初始化，warm-up，ADMM epoch loop（`g`-step、`z`-step、dual/`rho` update），phase-boundary `z` support masking，最后 raw-coordinate readout。

---

## 4. 评估指标说明

由于 polynomial expansion 会把原始特征映射到大量扩展特征，评估时需要把 expanded-space gate 聚合回 original feature space。

### 4.1 Legacy 结构化指标：paired-product score

paired-product score 的含义是：对每个原始特征，把它在原始项和相关 polynomial 项上的 gate 做乘法聚合，再用聚合分数排序原始特征。

直观上，它偏向选择“原始项和对应展开项都强”的特征。这个设计适合表达“原始特征和二次特征成组恢复”的假设。

它的风险也很明确：

- 它不是通用 feature importance 指标；
- 它自带 paired polynomial 假设；
- 在某些任务上可能高估 feature recovery；
- 因此不能单独作为论文的唯一 feature recovery 证据。

### 4.2 polynomial parent-sharing aggregation

degree-2 只用于 synthetic nonlinear benchmark。对于 expanded coordinate，更透明的 original-space readout 是 parent-sharing aggregation：把每个 polynomial term 的 score 按生成它的 original parent 分回去。

对 monomial：

```text
phi_a(x) = product_j x_j^{alpha_{a,j}}
P(a) = {j: alpha_{a,j} > 0}
```

如果 expanded feature `a` 的 score 是 `s_a`，则 original feature `j` 得到：

```text
S_j = sum_{a: j in P(a)} alpha_{a,j} / sum_l alpha_{a,l} * s_a
```

默认 score 使用 effective scale：

```text
s_a = |g_a * c_a|
```

直观例子：

| Expanded term | 分配规则 |
|---|---|
| `x_j` | 全部分给 `j` |
| `x_j^2` | 全部分给 `j` |
| `x_j x_k` | 一半给 `j`，一半给 `k` |
| `x_j^2 x_k` | `2/3` 给 `j`，`1/3` 给 `k` |

这个 readout 比 paired-product 更通用，因为它不假设“linear term 和 square term 必须同时强”。当前建议：**degree-2 synthetic 表里优先报告 parent-sharing gate score（`degree_share_g`）；如需和旧 effective-scale 结果对齐，同时保留 `degree_share_eff`、paired-product 作为 diagnostic。**

### 4.3 补充透明指标

论文应同时报告：

| 指标 | 含义 |
|---|---|
| `degree_share_eff` | 旧 effective-scale 诊断：按 monomial parent-sharing 聚合 `|g*c|` |
| `degree_share_g` | 按 monomial parent-sharing 聚合 raw gate `|g|` |
| `sum_eff` | 旧 effective-scale 诊断：对 `g * c` 取和 |
| `sum_g` | 对同一原始特征相关的 gate 取和 |
| `max_eff` | 旧 effective-scale 诊断：对 `g * c` 取最大值 |
| `L2_g` | 对同一原始特征相关的 gate 取 L2 norm |

推荐写法：

> 对 polynomial feature map，主 readout 使用 degree-normalized parent-sharing gate score；同时报告 paired-product、`degree_share_eff/sum_eff/sum_g/max_eff`，用于证明结论不依赖单一 readout 假设。

---

## 5. Clean 实验协议

本文把 clean experiment 定义为：一次只改变一个变量，其余主配置保持不变。

当前定下来的核心主方法：

```text
Model: GatedWeightNormMLP
First layer: column-normalized weight direction plus explicit scalar gate
Gate: linear / unbounded
Feature space: raw features or an explicit task feature map
Sparsity driver: ADMM + RatioNorm
Schedule: multi-phase medium-C schedule [0.1, 0.2, 0.5, 1.0, 2.0]
Warmup: per phase
Post-hoc pruning: ADMM-guided z-support masking at phase boundaries
Phase boundary: keep model weights and gate magnitudes; mask g by z support; reset z, u, Adam
Gate-rescaling: disabled
```

Synthetic nonlinear benchmark 的 instantiation：

```text
Feature map: full degree-2 polynomial expansion
Main readout: degree_share_g parent-sharing score
Required supporting readouts: paired-product, sum_eff, sum_g, max_eff, L2_g
```

主 synthetic 数据集：

```text
xor
ring
ring+xor
ring+xor+sum
```

Real-world 数据集来自 NIPS 2003 feature selection challenge：

| Dataset | 特征数 | 任务类型 | 备注 |
|---|---:|---|---|
| Madelon | 500 | synthetic / artificial classification | 含大量 distractor features，适合测试高维选择稳定性 |
| Arcene | 10,000 | cancer classification | mass-spectrometry 数据，样本少、维度高 |
| Gisette | 5,000 | handwritten digit classification | 高维视觉特征，预测任务较容易 |
| Dexter | 20,000 | text classification | sparse text features，高维稀疏输入 |

这些 real-world 数据主要用于验证 downstream predictive utility。除 Madelon 外，它们不提供和 synthetic benchmark 同等清晰的 feature ground truth，因此不应作为严格 feature recovery 证据。

主要报告：

```text
AUC: prediction performance
degree_share_g / parent-sharing score: default original-space recovery for polynomial inputs
paired-product score / group_product_current: legacy structured diagnostic
sum_g, max_g, L2_g: transparent original-space recovery
sum_eff, max_eff: effective-gate diagnostic when comparing older effective-scale readouts
```

---

## 6. Clean 实验关键结果

当前 synthetic instantiation（`GatedWeightNormMLP` + full degree-2 polynomial expansion + multi-phase ADMM + ADMM-guided `z` support masking）在 synthetic task 上的结果如下。注意：下面保留了 `mask_by_z` 和 `mask_by_z_final` 两类实验结果；paper-facing 方法写成统一的 phase-boundary `z` support masking，`mask_by_z_final` 作为强 ablation 结果解释。

结果文件：`results/ablation/phase_boundary_gate_update_20260502_114434.json`

| Phase-boundary update | C schedule | AUC | paired-product | `sum_g` | `sum_eff` | Alive gates |
|---|---|---:|---:|---:|---:|---:|
| Threshold on `g` | auto C | 0.786 | 1.000 | 0.573 | 0.594 | 147.0 |
| Mask `g` by `z` at every phase | [0.1, 0.2, 0.5, 1.0, 2.0] | 0.860 | 1.000 | **1.000** | **1.000** | 4.8 |
| **Mask `g` by `z` only at final phase** | **[0.1, 0.2, 0.5, 1.0, 2.0]** | **0.876** | **1.000** | **0.979** | **0.979** | **2.4** |

严格 seed-matched 复现实验仍支持同一方向，但数值略低：

结果文件：`results/ablation/phase_boundary_gate_update_20260502_122353.json`

| Phase-boundary update | C schedule | AUC | paired-product | `sum_g` | `sum_eff` | Alive gates |
|---|---|---:|---:|---:|---:|---:|
| **Mask `g` by `z`** | **[0.1, 0.2, 0.5, 1.0, 2.0]** | **0.833** | **1.000** | **0.958** | **0.948** | **3.2** |

因此 synthetic 主叙事可以从“threshold pruning 修正 approximate sparsity”升级为“ADMM 的 `z` support 提供离散选择，`g` 保留连续强度”。只在最后一个 phase 使用 `z` mask 的 AUC 最高，是一个重要诊断：过早锁死 support 有风险。主文中 paired-product 仍然应和 parent-sharing、`sum_g/sum_eff` 一起报告，因为后者是更透明的 original-space recovery 证据。

### 6.1 ADMM 是否优于 plain proximal gradient

结果文件：`results/ablation/multiphase_sparsity_driver_clean_20260501_140345.json`

| 方法 | AUC | paired-product | `sum_g` | `sum_eff` |
|---|---:|---:|---:|---:|
| **Multi-phase ADMM + RatioNorm** | **0.858** | **1.000** | **0.958** | **1.000** |
| Multi-phase prox + RatioNorm | 0.741 | **1.000** | 0.771 | 0.771 |

结论：在 multi-phase protocol 下，ADMM 是更好的 phase-internal sparsity driver。这个结论不等于“ADMM 在所有条件下都优于 prox”；single-run prox 曾表现较强，但不是当前主配置。

single-run 诊断结果如下：

结果文件：`results/ablation/a2_a3_plain_prox_l1_20260430_205052.json`

| Single-run 方法 | AUC | paired-product | `sum_g` | `sum_eff` |
|---|---:|---:|---:|---:|
| Single-run ADMM + RatioNorm | 0.650 | 0.066 | 0.295 | 0.406 |
| Single-run prox + RatioNorm | **0.786** | **1.000** | **0.799** | **0.812** |
| Single-run prox + plain L1 | 0.638 | **1.000** | 0.278 | 0.271 |

解读：single-run 下 prox + RatioNorm 是强 baseline，说明“稀疏驱动器”不能脱离 schedule 单独下结论。paper-facing 的 clean 结论应限定为：在相同 multi-phase protocol 下，ADMM + RatioNorm 优于 prox + RatioNorm。

### 6.2 phase-boundary pruning 是否必要

单 phase 诊断实验：`results/ablation/soft_pruning_necessity_D_20260427_181247.json`

| 配置 | AUC | best-k | strict zero count |
|---|---:|---:|---:|
| **Soft pruning** | **0.747** | **0.958** | **7955.5** |
| No pruning | 0.739 | 0.090 | 0.0 |

multi-phase aggregation 实验：`results/ablation/multiphase_aggregation_gap_20260428_170151.json`

| 配置 | AUC | paired-product | `sum_g` | `sum_eff` |
|---|---:|---:|---:|---:|
| **Multi-phase + soft pruning** | **0.860** | **1.000** | **0.944** | **0.965** |
| Multi-phase + no pruning | 0.860 | 0.337 | 0.715 | 0.760 |

这些早期诊断说明，phase boundary 上必须把近似稀疏 gate 转成严格支撑，否则 prediction AUC 可以接近，但 feature readout 会明显不稳定。

新的 phase-boundary ablation 进一步说明，最好的 boundary rule 不是直接阈值化 `g`，而是使用 `z` 的 support：

结果文件：`results/ablation/phase_boundary_gate_update_20260502_111513.json`

| Boundary rule | AUC | paired-product | `sum_g` | `sum_eff` | Alive gates | 解读 |
|---|---:|---:|---:|---:|---:|---|
| Threshold on `g` | 0.747 | 0.208 | 0.479 | 0.479 | 1892.0 | gate 仍偏分散 |
| Replace `g` with `z` | 0.732 | 1.000 | 0.812 | 0.812 | 1.1 | support 好，但幅度信息丢失 |
| **Mask `g` by `z`** | **0.781** | **1.000** | **0.875** | **0.875** | **3.0** | 使用 `z` support，同时保留 `g` magnitude |

补充正式 6-fold 结果显示，只在最后一个 phase 使用 `z` mask 更好：

结果文件：`results/ablation/phase_boundary_gate_update_20260503_165014.json`

| Boundary rule | C schedule | AUC | paired-product | `sum_g` | `sum_eff` |
|---|---|---:|---:|---:|---:|
| `mask_by_z_final` | [0.05, 0.1, 0.2, 0.5, 1.0] | 0.839 | 1.000 | 0.875 | 0.875 |
| **`mask_by_z_final`** | **[0.1, 0.2, 0.5, 1.0, 2.0]** | **0.876** | **1.000** | **0.979** | **0.979** |

结论：paper-facing 方法应写成 ADMM-guided `z` support masking，而不是普通 threshold soft pruning。`mask_by_z_final` 可以作为 synthetic 上表现最强的 boundary-calibration 变体；旧 threshold pruning 可以作为 ablation baseline。

### 6.3 polynomial readout：parent-sharing 是否比 paired-product 更稳妥

新增 parent-sharing aggregation 复跑：

结果文件：`results/ablation/phase_boundary_gate_update_20260503_205520.json`

| Readout / aggregation | Mean best-k |
|---|---:|
| paired-product / `group_product_current` | 1.000 |
| `degree_share_eff` | 0.896 |
| `degree_share_g` | 0.896 |
| `sum_eff` | 0.896 |
| `linear_only_eff` | 0.812 |

per-task `degree_share_eff`：

| Task | `degree_share_eff` | AUC |
|---|---:|---:|
| XOR | 1.000 | 0.991 |
| Ring | 0.583 | 0.561 |
| Ring+XOR | 1.000 | 0.766 |
| Ring+XOR+Sum | 1.000 | 0.696 |

结论：paired-product 对 synthetic score 明显更乐观；parent-sharing aggregation 更透明，也更适合作为 polynomial expanded-space 的默认 readout。这个结果进一步支持“degree-2 只是 synthetic feature map，不是方法本体”：在 fairer parent-sharing 下，degree-2 的 feature selection 分数不是满分，而 raw-feature `mask_by_z_final` 的 `sum_eff=0.938` 反而更能说明核心 support-learning mechanism。

### 6.4 gate_weight_norm 是否有效

结果文件：`results/ablation/c3_gate_weight_norm_clean_20260501_201224.json`

| Gate 设计 | AUC | paired-product | `sum_g` | `sum_eff` |
|---|---:|---:|---:|---:|
| Input gate | 0.769 | 0.931 | 0.535 | 0.566 |
| **gate_weight_norm** | **0.813** | **0.958** | **0.604** | **0.646** |

结论：重构前向后，`gate_weight_norm` 仍显著优于 input gate（AUC +0.044）并同步提升 paired-product 与 `sum_g/sum_eff`，可作为主方法的稳健组件之一。写作时强调它的作用是把 feature scale 显式放到 `g_j` 上，而不是让第一层 raw column norm 隐式承担 feature importance。

### 6.5 full degree-2 expansion 是否必要

结果文件：`results/ablation/synthetic_g3_diagonal_poly_20260430_193407.json`

| Feature space | 维度 | AUC | `sum_eff` |
|---|---:|---:|---:|
| No polynomial expansion, legacy auto-C threshold | 128 | 0.692 | 0.438 |
| **No polynomial expansion, `mask_by_z_final` medium-C** | **128** | **0.759** | **0.938** |
| **Full degree-2 expansion** | 8384 | **0.858** | **1.000** |
| Diagonal-only `[x_j, x_j^2]` | 256 | 0.762 | 0.615 |

新增 raw-feature 结果文件：`results/ablation/synthetic_g3_diagonal_poly_20260503_180656.json`

raw features + `mask_by_z_final` medium-C 的 per-task AUC：

| Task | AUC | `sum_eff` |
|---|---:|---:|
| XOR | 1.000 | 1.000 |
| Ring | 0.586 | 0.750 |
| Ring+XOR | 0.755 | 1.000 |
| Ring+XOR+Sum | 0.693 | 1.000 |

结论：ADMM `z`-masking 的 feature selection 机制不依赖 degree-2 expansion；不用 expansion 时，original-space recovery 仍然很强（overall `sum_eff=0.938`）。但 raw features 无法表达 Ring 的二次边界，导致 predictive AUC 明显下降。因此论文里应把 degree-2 写成 synthetic nonlinear benchmark 的 explicit feature map，而不是方法本体。diagonal-only expansion 维度可行，但已有结果只达到中间水平；它不能处理 cross-feature interactions，因此暂时不值得作为主线继续跑。

### 6.6 multi-phase 是否能被 single-run annealing 替代

结果文件：`results/ablation/e2_single_run_anneal_c_20260430_152059.json`

| Single-run schedule | AUC | paired-product | `sum_g` | `sum_eff` |
|---|---:|---:|---:|---:|
| Step schedule | 0.650 | 0.066 | 0.295 | 0.406 |
| Linear schedule | 0.628 | 0.104 | 0.271 | 0.361 |
| Log schedule | 0.629 | 0.111 | 0.274 | 0.372 |

结论：single-run annealing 不能替代 multi-phase。有效机制来自 phase boundary、per-phase warmup 和 ADMM-guided pruning 的组合。

### 6.7 跨 phase 保留 ADMM state 是否能替代 phase-boundary pruning

结果文件：`results/ablation/cross_phase_state_h_interaction_20260429_135829.json`

| 配置 | AUC | paired-product | `sum_g` | `sum_eff` |
|---|---:|---:|---:|---:|
| No-pruning baseline | 0.872 | 0.351 | 0.694 | 0.792 |
| Keep `z + u` | 0.872 | 0.288 | 0.663 | 0.760 |
| Keep `u + Adam` | 0.830 | 0.132 | 0.618 | 0.660 |
| Keep `z + u + Adam` | 0.701 | 0.139 | 0.347 | 0.389 |
| Keep `z + u + Adam` + soft pruning | 0.730 | 0.802 | 0.486 | 0.528 |

结论：保留 ADMM state 不能替代 phase-boundary pruning。这个替代假设不应作为论文主方案。

### 6.8 phase 之间 是否应该 reinitialize model weights

结果文件：`results/ablation/phase_boundary_weight_reinit_clean_20260501_152122.json`

| Weight policy | AUC | paired-product | `sum_g` | `sum_eff` |
|---|---:|---:|---:|---:|
| **Keep model weights** | **0.858** | **1.000** | **0.958** | **1.000** |
| Reinitialize model weights | 0.625 | 0.267 | 0.278 | 0.264 |

结论：phase 之间应保留 model weights。这里讨论的是模型权重是否重初始化，不是 pruning 后对 gate 数值做 rescaling。

另外测试过“只重置非零 `g`，不重置 weights”的诊断版本：它没有解决问题。quick 2-fold 结果中 AUC 只有 0.604，`sum_eff=0.406`，并且 alive gates 约 8322，说明只把非零 `g` reset 到 1 会破坏稀疏结构，不应作为主方法。

### 6.9 per-phase warmup 是否有用

结果文件：`results/ablation/warmup_per_phase_ablation_20260424_102914.json`

| Dataset | Warmup only phase 0 | **Warmup every phase** |
|---|---:|---:|
| XOR | **1.000** | **1.000** |
| Ring | 0.667 | **0.833** |
| Ring+XOR | 0.500 | **0.583** |
| Ring+XOR+Sum | **0.533** | **0.533** |

结论：per-phase warmup 对 Ring 和混合非线性任务有帮助，应保留为 multi-phase mechanism 的一部分。

---

## 7. 外部 benchmark

### 7.1 Synthetic benchmark


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
| **GRN-FS** | Embedded (DL) | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | **0.8580** |
| Saliency | Attribution | 0.33 | 0.00 | 0.08 | 0.36 | 0.1944 | N/A |
| nn | Attribution | 0.89 | 0.85 | 0.93 | 0.94 | 0.9022 | 0.5250 |
| GuidedBackprop | Attribution | 0.33 | 0.00 | 0.08 | 0.36 | 0.1944 | N/A |
| Deconvolution | Attribution | 0.33 | 0.00 | 0.00 | 0.36 | 0.1736 | N/A |
| InputXGradient | Attribution | 0.25 | 0.08 | 0.04 | 0.36 | 0.1840 | N/A |
| IG_noMul | Attribution | 0.25 | 0.08 | 0.04 | 0.36 | 0.1840 | N/A |
| SmoothGrad | Attribution | 0.33 | 0.08 | 0.04 | 0.39 | 0.2118 | N/A |
| DeepLift | Attribution | 0.25 | 0.08 | 0.04 | 0.36 | 0.1840 | N/A |
| FeatureAblation | Attribution | 0.33 | 0.00 | 0.08 | 0.39 | 0.2014 | N/A |
| FeaturePermutation | Attribution | 0.25 | 0.00 | 0.04 | 0.36 | 0.1632 | N/A |
| ShapleyValueSampling | Attribution | 0.08 | 0.08 | 0.04 | 0.39 | 0.1493 | N/A |

这张表里的 GRN-FS 是 legacy paired-product / older run 结果，不应再作为唯一主证据。当前更透明的 polynomial parent-sharing rerun 是：paired-product 1.000，但 `degree_share_eff=0.896`、`sum_eff=0.896`。因此 synthetic degree-2 应写成 optional interaction-dictionary ablation，而不是主方法本体；主文必须同时报告 parent-sharing 和 paired-product。

### 7.2 Real-world benchmark

当前 real-world NIPS 2003 结果支持 GRN-FS 的 downstream predictive utility。论文主表应使用 benchmark 的 fixed-budget protocol，而不是 retrospective best-$k$ sweep：

- Madelon: `k=20`
- Arcene: `k=7000`
- Gisette: `k=3500`
- Dexter: `k=10000`

`Mean ± std` 是四个数据集之间的 task-level dispersion，不是每个 baseline 的 seed std。原因是 benchmark baseline 文件大多只有单次固定配置结果；GRN-FS/STG 有 seed std，但为了表格可比性，主表统一报告 across-dataset std。mRMR 缺 Dexter 文件，因此 mean/std 只基于 3 个 available datasets。

| Method | Madelon | Arcene | Gisette | Dexter | Mean ± std |
|---|---:|---:|---:|---:|---:|
| GRN-FS | 0.964 | 0.890 | 0.995 | 0.973 | 0.956 ± 0.046 |
| RF | **0.965** | **0.906** | 0.995 | 0.977 | **0.961 ± 0.038** |
| TreeSHAP | 0.964 | 0.901 | 0.995 | 0.980 | 0.960 ± 0.041 |
| ReliefF | 0.941 | 0.892 | 0.995 | 0.976 | 0.951 ± 0.045 |
| MI | 0.833 | 0.885 | 0.995 | 0.982 | 0.924 ± 0.078 |
| mRMR | 0.640 | 0.884 | 0.995 | -- | 0.840 ± 0.181 |
| LassoNet | 0.950 | 0.890 | 0.995 | 0.979 | 0.954 ± 0.046 |
| STG | 0.542 | 0.891 | 0.995 | 0.978 | 0.852 ± 0.211 |
| CAE | 0.711 | 0.878 | 0.995 | 0.979 | 0.891 ± 0.130 |
| FSNet | 0.752 | 0.898 | 0.994 | 0.903 | 0.887 ± 0.100 |
| DeepPINK | 0.742 | 0.887 | 0.995 | 0.975 | 0.900 ± 0.115 |
| CancelOut-sigmoid | 0.697 | 0.886 | 0.995 | 0.979 | 0.889 ± 0.137 |
| CancelOut-softmax | 0.740 | 0.891 | 0.994 | 0.954 | 0.895 ± 0.111 |
| Saliency | 0.630 | 0.879 | 0.995 | 0.980 | 0.871 ± 0.169 |
| Input×Gradient | 0.657 | 0.898 | 0.995 | **0.982** | 0.883 ± 0.157 |
| Integrated Gradients | 0.635 | 0.895 | 0.995 | 0.977 | 0.876 ± 0.166 |
| SmoothGrad | 0.640 | 0.894 | 0.995 | 0.979 | 0.877 ± 0.164 |
| Guided Backprop | 0.630 | 0.879 | 0.995 | 0.980 | 0.871 ± 0.169 |
| DeepLIFT | 0.636 | 0.887 | 0.995 | 0.981 | 0.875 ± 0.166 |
| Deconvolution | 0.630 | 0.879 | 0.995 | 0.980 | 0.871 ± 0.169 |
| Feature Ablation | 0.658 | 0.893 | **0.995** | 0.982 | 0.882 ± 0.156 |
| Feature Permutation | 0.623 | 0.899 | 0.995 | 0.981 | 0.875 ± 0.173 |
| Shapley sampling | 0.639 | 0.893 | 0.995 | 0.980 | 0.877 ± 0.165 |
| Random ranking | 0.612 | 0.889 | 0.995 | 0.972 | 0.867 ± 0.176 |

结论：real-world 结果说明方法有预测可用性，并且在 fixed-budget NIPS 2003 protocol 下处于 RF/TreeSHAP aggregate tier、略高于 LassoNet、明显高于 STG。但这些数据集没有 synthetic benchmark 那样清晰的 feature ground truth，因此不应被过度解释为严格恢复证据。

### 7.3 DAG benchmark

DAG benchmark 上，`z` masking 没有成为更好的默认选择。它可以保持接近的 predictive AUC，但 support 往往过窄，结构恢复指标低于旧的 low-C threshold setting。

| Setting | C schedule | AUC | AUPRC | Alive gates | bestK2 | top20 chain/fork |
|---|---|---:|---:|---:|---:|---:|
| Threshold on `g` | [0.1, 0.2, 0.3, 0.4, 0.5] | **0.8200** | **0.8076** | 84.7 | **0.0823** | **3.83** |
| `mask_by_z` | [0.1, 0.2, 0.3, 0.4, 0.5] | 0.8156 | 0.8006 | 3.7 | 0.0617 | 2.33 |
| `mask_by_z_final` | [0.1, 0.2, 0.3, 0.4, 0.5] | 0.8136 | 0.8016 | 3.7 | 0.0617 | 2.50 |
| `mask_by_z` | [0.001, 0.005, 0.01, 0.02, 0.05] | 0.7211 | 0.7205 | 274.3 | 0.0720 | 3.67 |
| `mask_by_z_final` | [0.001, 0.005, 0.01, 0.02, 0.05] | 0.7107 | 0.7159 | 295.0 | 0.0720 | 3.83 |

结果文件：

| Setting | 文件 |
|---|---|
| Threshold low-C baseline | `results/ablation/dag_lowc_attribution_benchmark_20260430_174900.json` |
| `mask_by_z` low-C | `results/ablation/dag_mask_by_z_low_benchmark_20260502_135848.json` |
| `mask_by_z_final` low-C | `results/ablation/dag_mask-by-z-final_low_benchmark_20260502_142314.json` |
| `mask_by_z` tiny-C | `results/ablation/dag_mask-by-z_tiny_benchmark_20260502_142008.json` |
| `mask_by_z_final` tiny-C | `results/ablation/dag_mask-by-z-final_tiny_benchmark_20260502_142625.json` |

结论：synthetic 上 `z` support 是合适的离散选择信号；DAG 上直接用 `z != 0` 太离散，旧的 threshold low-C 方案更稳。论文里可以把 DAG 写成外部分布下的边界条件：同一 ADMM mechanism 在不同 feature graph 结构上需要不同的 boundary calibration。

---

## 8. 当前方法定位

先不定 paper wording，先定方法定位。当前共识是：GRN-FS 的主方法不是 degree-2 expansion，也不是单独的 `z != 0` mask，而是一个 staged sparse-support learning lifecycle，其中 gate 是 input-dimension weight-normalization 的 scalar scale。

核心主方法：

```text
GRN-FS learns input-dimension weight-normalized gates with a neural model,
sparsifies them through ADMM, stabilizes the support across phases, and
converts continuous gates into a strict selected feature set by masking g with
the ADMM support z.
```

主方法包含四个必要组件：

| 组件 | 当前选择 | 作用 |
|---|---|---|
| Predictor gate | `GatedWeightNormMLP` | 用 normalized input-column direction + scalar gate 学任务相关的 feature strength |
| Sparse driver | ADMM + RatioNorm | 把 sparsity 放到 `z` 的 proximal update 上 |
| Training lifecycle | multi-phase + per-phase warmup + keep weights | 逐步增强稀疏压力，同时保留已学 representation |
| Support conversion | `mask_by_z` / ADMM support masking | 用 `z != 0` 决定 support，保留 retained `g` magnitude |

feature map 的定位：

- Raw features 是最干净的 mechanism check：raw-feature `mask_by_z_final` medium-C 下 `sum_eff=0.938`，说明 selection mechanism 本身有效；这条结果作为 ablation 支撑，不把 final-only 写成唯一主方法。
- Full degree-2 是 synthetic nonlinear benchmark 的 explicit feature map：它把 AUC 从 raw 的 0.759 提到 0.876，但不应被描述成通用必要组件。
- Diagonal-only 暂时不进主线：维度可行，但已有结果只是中间水平，且不能表达 cross-feature interactions。

readout 的定位：

- raw-feature setting 默认用 `|g|` 排序；`|g*c|` 属于 older effective-scale diagnostic，不作为 weight-normalized gate 版本的核心定义；
- degree-2 setting 需要 original-space aggregation；
- paired-product 可以作为 structured readout，但必须同时报告 `sum_eff/sum_g/max_eff/L2_g`。

不推荐的替代写法：

```text
Soft pruning can be removed by preserving cross-phase ADMM state.
```

这个假设已经被实验否定。

也不推荐写：

```text
z masking uniformly improves all benchmarks.
```

DAG benchmark 不支持这个说法。

---

## 9. 写作边界

### 可以写

- 主协议下，ADMM 是更好的 phase-internal sparsity driver。
- ADMM-guided `z` support masking 是从 ADMM proximal support 到 strict gate readout 的桥接步骤。
- `gate_weight_norm` 是有效架构改进；它把 feature scale 显式放到 `g_j`，提高 gate 作为选择变量的可信度。
- full degree-2 expansion 是 synthetic nonlinear benchmark 的 optional feature map；raw-feature ablation 说明 selection mechanism 本身仍有效。
- phase boundary 应保留 model weights。
- synthetic benchmark 上，`mask_by_z_final` medium-C 是效果很强的 boundary-calibration 变体，但 paper-facing algorithm 写成统一的 `mask_by_z` support masking。
- DAG benchmark 上，旧 low-C threshold rule 仍是更稳的外部基线。

### 不应写

- 不要说 paired-product score 单独证明 feature recovery。
- 不要说 `z` masking 对所有 benchmark 都最好；DAG 结果不支持。
- 不要说 ADMM 总是优于 prox；clean 结论限定在主 multi-phase protocol 下。
- 不要把 gate-rescaling reweighting 当成 reinitialize model weights。
- 不要把 `g <- z` 和 `g <- g * 1[z != 0]` 混为一谈；实验支持的是后者。
- 不要把 degree-2 expansion 写成通用必要组件；高维 real-world / DAG 不适合 full degree-2 expansion。
- 不要把主方法简化成 `z != 0` mask；它依赖 weight-normalized gate、ADMM、multi-phase、warmup、keep weights 和 z-support masking 的组合。

---

## 10. 结果文件索引

### Clean / paper-facing 结果

| 主题 | 文件 |
|---|---|
| ADMM vs prox | `results/ablation/multiphase_sparsity_driver_clean_20260501_140345.json` |
| input gate vs `gate_weight_norm` | `results/ablation/c3_gate_weight_norm_clean_20260501_201224.json` |
| keep vs reinitialize model weights | `results/ablation/phase_boundary_weight_reinit_clean_20260501_152122.json` |
| diagonal-only polynomial | `results/ablation/synthetic_g3_diagonal_poly_20260430_193407.json` |
| raw features + final z-mask medium-C | `results/ablation/synthetic_g3_diagonal_poly_20260503_180656.json` |
| full degree-2 + parent-sharing metric diagnostic | `results/ablation/synthetic_g3_diagonal_poly_20260503_203124.json` |
| multi-phase soft vs no pruning | `results/ablation/multiphase_aggregation_gap_20260428_170151.json` |
| cross-phase state interaction | `results/ablation/cross_phase_state_h_interaction_20260429_135829.json` |
| single-run C annealing | `results/ablation/e2_single_run_anneal_c_20260430_152059.json` |
| gate-rescaling internal result | `results/ablation/reweight_modes_r2_r3_20260430_211241.json` |
| phase-boundary z-mask main result | `results/ablation/phase_boundary_gate_update_20260502_114434.json` |
| phase-boundary z-mask seed-matched result | `results/ablation/phase_boundary_gate_update_20260502_122353.json` |
| final-only z-mask diagnostic | `results/ablation/phase_boundary_gate_update_20260503_165014.json` |
| polynomial parent-sharing readout rerun | `results/ablation/phase_boundary_gate_update_20260503_205520.json` |

### Benchmark 结果

| 主题 | 文件 |
|---|---|
| Synthetic degree-2 main benchmark | `results/main/best_gradual_poly2_benchmark_20260424_181312.json` |
| Selection comparison | `results/main/selection_comparison_20260425_165507.json` |
| Gate weight norm comparison | `results/main/gate_weight_norm_comparison_20260425_213958.json` |
| DAG threshold low-C baseline | `results/ablation/dag_lowc_attribution_benchmark_20260430_174900.json` |
| DAG z-mask low-C | `results/ablation/dag_mask_by_z_low_benchmark_20260502_135848.json` |
| DAG z-mask final/tiny-C | `results/ablation/dag_mask-by-z-final_tiny_benchmark_20260502_142625.json` |

---

## 11. 代码索引

| 组件 | 文件 |
|---|---|
| ADMM core training | `src/admm_input_group_wrapper.py` |
| Gradual ADMM and pruning | `src/gradual_admm_with_pruning.py` |
| Polynomial expansion | `src/polynomial_expansion.py` |
| original-space aggregation / parent-sharing readout | `experiments/ablation/run_cross_phase_state_h_main.py` |
| phase-boundary z-mask ablation | `experiments/ablation/run_phase_boundary_gate_update_ablation.py` |
| ADMM vs prox clean experiment | `experiments/ablation/run_multiphase_sparsity_driver_clean.py` |
| gate_weight_norm clean experiment | `experiments/ablation/run_c3_gate_weight_norm_clean.py` |
| weight reinitialization clean experiment | `experiments/ablation/run_phase_boundary_weight_reinit_clean.py` |
| feature-space ablation: raw / full degree-2 / diagonal | `experiments/ablation/run_synthetic_g3_diagonal_poly.py` |
| multi-phase aggregation gap | `experiments/ablation/run_multiphase_aggregation_gap.py` |
| cross-phase state experiments | `experiments/ablation/run_cross_phase_state_h_main.py`, `experiments/ablation/run_cross_phase_state_h_interaction.py` |
| DAG z-mask benchmark | `experiments/main/run_dag_mask_by_z_benchmark.py` |

---

## 12. 当前推荐主配置

核心主方法配置：

```text
Model: GatedWeightNormMLP
First layer: column-normalized weight direction plus explicit scalar gate
Gate: linear / unbounded
Feature space: raw features or explicit task feature map
Sparsity driver: ADMM + RatioNorm
Schedule: multi-phase medium-C schedule [0.1, 0.2, 0.5, 1.0, 2.0]
Warmup: per phase
Post-hoc pruning: ADMM-guided z-support masking
Phase boundary: keep model weights and gate magnitudes; mask g by z support; reset z, u, Adam
Gate-rescaling: disabled
```

Synthetic nonlinear benchmark instantiation：

```text
Feature space: full degree-2 polynomial expansion
Main readout: degree_share_g parent-sharing score
Diagnostic readouts: paired-product, sum_eff, sum_g, max_eff, L2_g
```

Raw-feature ablation instantiation：

```text
Feature space: original raw features
Main readout: |g|
Purpose: show the support-learning mechanism does not rely on degree-2 expansion
```

当前 paper-facing algorithm 使用 `soft_prune_update="mask_by_z"` 的 support-masking 叙事；`mask_by_z_final` + medium-C schedule `[0.1, 0.2, 0.5, 1.0, 2.0]` 是 synthetic 上表现很强的 boundary-calibration 变体。DAG benchmark 当前不放主线；如需报告，应作为 appendix / limitation，说明 high-dimensional external tasks 需要 boundary calibration，不把 z-mask 写成 universal improvement。

最终方法不应被描述成单一 trick。它的效果来自 weight-normalized scalar gate、ADMM + RatioNorm sparse driver、multi-phase warmup/keep-weights lifecycle、ADMM-guided `z` support masking，以及透明的 readout/aggregation 共同作用。Degree-2 expansion 应被描述为 synthetic nonlinear benchmark 上使用的 explicit feature map，而不是 GRN-FS 的通用必要组件。
