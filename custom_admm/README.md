# ADMM Input Group - Feature Selection Benchmark

本项目实现了基于 ADMM（Alternating Direction Method of Multipliers）+ Ratio Norm 的特征选择方法，
并通过标准的 Feature Selection Benchmark 进行评估。

## 项目结构

```
NEW_Pruning_20251110/
├── custom_admm/                      # 我们的自定义 ADMM 实现
│   ├── run_admm_input_group_benchmark.py  # 主 benchmark 脚本
│   ├── src/
│   │   ├── admm_input_group_wrapper.py    # ADMM Input Group 实现
│   │   ├── data.py                        # 数据生成（与 benchmark 一致）
│   │   ├── core.py                        # 评估核心逻辑
│   │   ├── nn_wrapper.py                  # 神经网络包装器
│   │   └── utils.py                       # 工具函数
│   └── results/                           # 输出结果
│
├── Feature-Selection-Benchmark/      # 官方 Benchmark（不修改）
│   ├── main-benchmark.py
│   ├── src/
│   │   ├── data.py
│   │   └── core.py
│   └── results/
│
└── results_backup/                   # 历史结果备份
    ├── admm_input_group-*.txt
    └── cae-*.txt
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

### 运行 Benchmark

```bash
cd custom_admm
python run_admm_input_group_benchmark.py
```

### 输出结果

结果保存在 `custom_admm/results/` 目录下：

- `admm_input_group-xor-1000.txt`
- `admm_input_group-ring-1000.txt`
- `admm_input_group-ring+xor-1000.txt`
- `admm_input_group-ring+xor+sum-1000.txt`

### 结果格式

```
Dataset    ADMM_InputGroup_bestK    ADMM_InputGroup_best2K    ...
xor_2_1000    1.0    1.0
xor_4_1000    1.0    1.0
...
```

## 数据集

| 数据集 | k (true features) | 维度范围 |
|--------|------------------|---------|
| xor | 2 | 2-2048 |
| ring | 2 | 8-2048 |
| ring+xor | 4 | 4-2048 |
| ring+xor+sum | 6 | 6-2048 |

## 评估指标

- **best-k**: 前 k 个选中特征中 true features 的比例
- **best-2k**: 前 2k 个选中特征中 true features 的比例

## 与官方 Benchmark 的比较

### 相同点

1. 使用相同的数据生成函数 (`data.py`)
2. 使用相同的 6 折交叉验证
3. 使用相同的特征洗牌协议
4. 使用相同的评估指标

### 不同点

1. **特征选择方法**: 我们用 ADMM Input Group，官方 benchmark 支持 CAE、DeepPINK 等
2. **代码位置**: 我们在 `custom_admm/` 中独立运行

## 算法核心

ADMM Input Group 的核心思想：

1. **输入层门控**: 为每个特征学习一个 gate 值 `g`
2. **Ratio Norm 正则化**: `R(z) = ||z||₁ / ||z||₂`（尺度不变）
3. **ADMM 优化**: 将问题分解为 g-step（梯度下降）和 z-step（Ratio Norm proximal）

详细算法推导请参考论文/技术报告。

## 历史结果

历史运行结果保存在 `results_backup/` 目录，可用于对比。

## 依赖

```
torch>=1.8
numpy>=1.19
scikit-learn>=0.24
```

## 开发注意事项

1. **不要修改 Feature-Selection-Benchmark/** - 这是官方参考
2. **数据生成功能必须与 benchmark 一致** - `custom_admm/src/data.py` 应与官方版本同步
3. **结果格式应与官方兼容** - 便于后续对比分析

## 故障排除

### 问题：导入错误 "No module named 'data'"

解决：确保从 `custom_admm/` 目录运行脚本，或添加路径：
```python
import sys
sys.path.insert(0, '/path/to/custom_admm')
```

### 问题：结果与历史结果不一致

可能原因：
1. 数据生成逻辑变化
2. 随机种子不同
3. 特征洗牌顺序不同

检查 `custom_admm/src/data.py` 是否与官方版本一致。
