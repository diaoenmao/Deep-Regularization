# Phase 1 实验记录 - 早停、噪声、优化器对比

## 实验目标

快速验证以下改进对 ADMM Input Group 性能的影响：

1. **早停机制 (Early Stopping)** - 防止过拟合
2. **高斯噪声 (Gaussian Noise)** - 增加正则化
3. **优化器对比 (Adagrad vs Adam)** - 测试 benchmark 默认优化器

## 实验配置

### 数据集配置

| 数据集 | k (true features) | 测试维度 (quick mode) |
|--------|------------------|----------------------|
| XOR | 2 | [8, 128, 1024] |
| RING | 2 | [32, 512] |
| RING+XOR | 4 | [16, 256] |
| RING+XOR+SUM | 6 | [64] |

### 实验组配置

| 实验名 | 高斯噪声 | 早停 | 优化器 | 说明 |
|--------|---------|------|--------|------|
| `baseline` | 0.0 | ✗ | Adam | 原始 ADMM |
| `early_stop` | 0.0 | ✓ (patience=66) | Adam | + 早停 |
| `noise` | 0.747 | ✗ | Adam | + 高斯噪声 |
| `early_stop_noise` | 0.747 | ✓ (patience=66) | Adam | + 早停 + 噪声 |
| `adagrad` | 0.0 | ✗ | Adagrad | + Adagrad 优化器 |
| `full_improvement` | 0.747 | ✓ (patience=66) | Adagrad | 全部改进 |

### 固定超参数

```python
lr = 0.005
C = 0.05
batch_size = 64
warmup_epochs = 120
max_epochs = 500
feat_drop = 0.6
dropout = 0.043
latent_size = 58
n_hidden_layers = 5
activation = 'mish'
```

## 使用方法

### 快速模式 (测试部分维度)

```bash
cd custom_admm
python run_phase1_experiments.py --quick --gpu 0
```

### 完整模式 (测试所有维度)

```bash
python run_phase1_experiments.py --gpu 0
```

## 输出文件

- `results/phase1_experiments_YYYYMMDD_HHMMSS.json` - 完整结果
- `results/phase1_summary_YYYYMMDD_HHMMSS.txt` - 可读摘要

## 代码修改

### `src/admm_input_group_wrapper.py`

新增参数：
```python
optimizer_type: str = "adam"  # "adam" or "adagrad"
use_early_stopping: bool = False
patience: int = 66
val_split: float = 0.2
```

早停逻辑：
- 每个 ADMM epoch 后计算验证损失
- 保存最佳状态
- 无改善达 patience 个 epoch 后停止
- 恢复最佳状态

## 预期结果

| 改进 | 预计 RING 提升 | 预计整体提升 |
|------|--------------|------------|
| 早停 | +5-10% | +3-5% |
| 高斯噪声 | +3-5% | +2-3% |
| Adagrad | +2-5% | +1-3% |
| 全部 | +10-20% | +6-11% |

## 实验进度

- [ ] 运行 baseline
- [ ] 运行 early_stop
- [ ] 运行 noise
- [ ] 运行 early_stop_noise
- [ ] 运行 adagrad
- [ ] 运行 full_improvement
- [ ] 生成对比报告

## 备注

- Quick mode 约需 30-60 分钟 (GPU: RTX A4000)
- Full mode 约需 2-4 小时
