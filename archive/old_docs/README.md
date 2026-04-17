# SADMM-FS: Sparse ADMM Feature Selection

**最后更新**: 2026-04-10

本项目实现了 **SADMM-FS** (Scope-Driven Stochastic ADMM for Feature Selection)，基于 ADMM + Ratio Norm 的神经特征选择方法。

---

## 核心方法

**SADMM-FS** 核心组件:
1. 全局门控向量 g ∈ R^m (线性、无界)
2. ADMM 优化框架解耦梯度更新和稀疏化
3. 支持 Ratio Norm (L1/L2) 和标准 L1 惩罚
4. 特征 dropout (p=0.6-0.7)

---

## 目录结构

```
custom_admm/
├── src/                              # 核心实现
│   ├── admm_input_group_wrapper.py   # 主方法: SADMM-FS
│   ├── admm_utils.py                 # ADMM 求解器 (三次方程)
│   ├── nn_wrapper.py                 # 标准 MLP
│   ├── stg_wrapper.py                # STG baseline
│   ├── tabnet_wrapper.py             # TabNet baseline
│   ├── cae_wrapper.py                # CAE baseline
│   ├── e2efs_wrapper.py              # E2E-FS baseline
│   ├── mentor_models.py              # Transformer/Gate 变体
│   ├── iterative_run.py              # 迭代特征选择
│   ├── polynomial_expansion.py       # 多项式扩展
│   ├── data.py                       # 数据生成
│   └── utils.py                      # 工具函数
├── results/                          # 实验结果 (见 results/README.md)
├── experiments/                      # 实验脚本
└── run_*.py                          # 运行脚本
```

## 架构说明

### 设计原则

1. **Feature-Selection-Benchmark 文件夹保持不变** - 作为官方参考实现
2. **custom_admm 是独立副本** - 包含我们的自定义方法，可以独立运行
3. **数据生成保持一致** - `custom_admm/src/data.py` 与官方版本完全相同

### 数据流

```
generate_dataset() [from data.py]
        ↓
    X, X_tilde, y
        ↓
run_admm_input_group() [from admm_input_group_wrapper.py]
        ↓
    y_train_hat, y_hat, scores, scores2
        ↓
    best-k, best-2k metrics
```

## 使用方法

### 主实验

```bash
# Backbone + Tier-2 baseline
python run_admm_input_group_benchmark.py

# Matched neural baseline
python experiments/matched_neural_synthetic.py
```

### 消融实验

```bash
# 迭代特征选择
python run_iterative_ablation.py

# 多项式扩展
python run_polynomial_ablation.py

# Transformer pretrain
python run_transformer_pretrain_ablation.py

# Gating 消融
python run_bounded_gate_ablation.py
```

### Real-world 数据集

```bash
# NIPS 2003 数据集
python run_realworld_admm.py

# Modern 数据集
python run_modern_admm_stg.py
```

---

## 默认配置

```python
DEFAULT_CONFIG = {
    # Architecture
    "latent_size": 32,
    "n_hidden_layers": 2,
    "dropout": 0.043,
    "activation": "mish",
    "feat_drop": 0.6,

    # Training
    "epochs": 416,
    "warmup_epochs": 100,
    "optimizer": "adagrad",
    "lr": 0.00176,
    "batch_size": 56,
    "patience": 66,

    # ADMM
    "sparsity_penalty": "ratio_norm",  # or "l1"
    "C": 0.05,
}
```

---

## 关键结果

| 方法 | Mean best-k | Mean AUC |
|------|-------------|----------|
| **SADMM-FS** | **0.6278** | **0.6605** |
| TabNet | 0.2851 | 0.5995 |
| CAE | 0.1861 | 0.5236 |
| FSNet | 0.1667 | 0.5621 |

详细结果见 [results/README.md](results/README.md)

---

## Baseline 实现

| 方法 | 文件 | 描述 |
|------|------|------|
| STG | `src/stg_wrapper.py` | 官方 stg 包封装 |
| TabNet | `src/tabnet_wrapper.py` | pytorch_tabnet 封装 |
| CAE | `src/cae_wrapper.py` | Concrete Autoencoder |
| E2E-FS | `src/e2efs_wrapper.py` | End-to-End FS |
| CancelOut | `src/cancelout.py` | CancelOut layer |
| DeepPINK | `src/deeppink.py` | DeepPINK knockoff |
| FSNet | `src/fsnet.py` | FSNet selector |

---

## 数据集

| 数据集 | k (true features) | 描述 |
|--------|------------------|------|
| xor | 2 | XOR 交互检测 |
| ring | 2 | 圆形边界 |
| ring+xor | 4 | 混合任务 |
| ring+xor+sum | 4 | 混合 + 求和 |

Real-world: madelon, gisette, arcene, dexter (NIPS 2003)

---

## 评估指标

- **best-k**: 前 k 个选中特征中 true features 的比例
- **best-2k**: 前 2k 个选中特征中 true features 的比例
- **AUC**: ROC 曲线下面积
- **AUPRC**: PR 曲线下面积

---

## 依赖

```
torch>=1.8
numpy>=1.19
scikit-learn>=0.24
```
