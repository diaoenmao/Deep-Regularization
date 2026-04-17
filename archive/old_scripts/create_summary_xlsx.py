# -*- coding: utf-8 -*-
"""
Create comprehensive summary of all methods and results.
"""

import pandas as pd
from pathlib import Path

# ============================================================
# 1. Main Methods Comparison (Synthetic)
# ============================================================
main_methods = [
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'SADMM-FS (gated_mlp)', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.6278, 'Std': '-', 'Notes': 'Best overall'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'SADMM-FS (gated_mlp)', 'Dataset': 'Overall', 'Metric': 'AUC', 'Value': 0.6605, 'Std': '-', 'Notes': '-'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'STG', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.625, 'Std': '-', 'Notes': 'Backbone only'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'CancelOut', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.625, 'Std': '-', 'Notes': 'Full match'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'TabNet', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.2851, 'Std': '-', 'Notes': 'Standard impl'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'CAE', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.1861, 'Std': '-', 'Notes': 'Standard impl'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'E2E-FS', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.1767, 'Std': '-', 'Notes': 'Standard impl'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'FSNet', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.1667, 'Std': '-', 'Notes': 'Standard impl'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'DeepPINK', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.0938, 'Std': '-', 'Notes': 'Full match'},
    # Per-task results
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'SADMM-FS', 'Dataset': 'XOR (m=128)', 'Metric': 'best-k', 'Value': 1.000, 'Std': '-', 'Notes': 'Perfect'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'SADMM-FS', 'Dataset': 'Ring (m=128)', 'Metric': 'best-k', 'Value': 0.500, 'Std': '-', 'Notes': '-'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'SADMM-FS', 'Dataset': 'Ring+XOR (m=256)', 'Metric': 'best-k', 'Value': 0.625, 'Std': '-', 'Notes': '-'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'SADMM-FS', 'Dataset': 'Ring+XOR+Sum (m=256)', 'Metric': 'best-k', 'Value': 0.667, 'Std': '-', 'Notes': '-'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'STG', 'Dataset': 'XOR (m=128)', 'Metric': 'best-k', 'Value': 1.000, 'Std': '-', 'Notes': '-'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'STG', 'Dataset': 'Ring (m=128)', 'Metric': 'best-k', 'Value': 0.000, 'Std': '-', 'Notes': 'Failed'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'TreeSHAP', 'Dataset': 'Ring (m=128)', 'Metric': 'best-k', 'Value': 0.990, 'Std': '-', 'Notes': 'Best for Ring'},
    {'Category': 'Main Methods', 'Experiment': 'Synthetic Benchmark', 'Method': 'RF', 'Dataset': 'Ring (m=128)', 'Metric': 'best-k', 'Value': 1.000, 'Std': '-', 'Notes': 'Best for Ring'},
]

# ============================================================
# 2. Real-World Datasets
# ============================================================
realworld = [
    {'Category': 'Real-World', 'Experiment': 'NIPS 2003', 'Method': 'SADMM-FS', 'Dataset': 'madelon (m=500)', 'Metric': 'AUROC', 'Value': 0.965, 'Std': '-', 'Notes': 'Best'},
    {'Category': 'Real-World', 'Experiment': 'NIPS 2003', 'Method': 'SADMM-FS', 'Dataset': 'gisette (m=5000)', 'Metric': 'AUROC', 'Value': 0.985, 'Std': '-', 'Notes': 'Best'},
    {'Category': 'Real-World', 'Experiment': 'NIPS 2003', 'Method': 'SADMM-FS', 'Dataset': 'arcene (m=10000)', 'Metric': 'AUROC', 'Value': 0.887, 'Std': '-', 'Notes': 'Best'},
    {'Category': 'Real-World', 'Experiment': 'NIPS 2003', 'Method': 'SADMM-FS', 'Dataset': 'dexter (m=20000)', 'Metric': 'AUROC', 'Value': 0.889, 'Std': '-', 'Notes': 'Best'},
    {'Category': 'Real-World', 'Experiment': 'NIPS 2003', 'Method': 'STG', 'Dataset': 'madelon', 'Metric': 'AUROC', 'Value': 0.847, 'Std': '-', 'Notes': '-'},
    {'Category': 'Real-World', 'Experiment': 'NIPS 2003', 'Method': 'STG', 'Dataset': 'gisette', 'Metric': 'AUROC', 'Value': 0.963, 'Std': '-', 'Notes': '-'},
    {'Category': 'Real-World', 'Experiment': 'NIPS 2003', 'Method': 'STG', 'Dataset': 'arcene', 'Metric': 'AUROC', 'Value': 0.808, 'Std': '-', 'Notes': '-'},
    {'Category': 'Real-World', 'Experiment': 'NIPS 2003', 'Method': 'STG', 'Dataset': 'dexter', 'Metric': 'AUROC', 'Value': 0.825, 'Std': '-', 'Notes': '-'},
]

# ============================================================
# 3. Ablation: Iterative FS
# ============================================================
iterative = [
    {'Category': 'Ablation', 'Experiment': 'Iterative FS', 'Method': 'single_pass', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.0', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Iterative FS', 'Method': 'single_pass', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.67, 'Std': '0.24', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Iterative FS', 'Method': 'single_pass', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 0.67, 'Std': '0.12', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Iterative FS', 'Method': 'iterative_hard', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 0.67, 'Std': '0.24', 'Notes': 'Hard prune hurts'},
    {'Category': 'Ablation', 'Experiment': 'Iterative FS', 'Method': 'iterative_hard', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.58, 'Std': '0.34', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Iterative FS', 'Method': 'lottery_ticket', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.08, 'Std': '0.19', 'Notes': 'Failed!'},
    {'Category': 'Ablation', 'Experiment': 'Iterative FS', 'Method': 'gradual_admm', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.0', 'Notes': 'Best'},
    {'Category': 'Ablation', 'Experiment': 'Iterative FS', 'Method': 'gradual_admm', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.0', 'Notes': 'Best: +33%'},
]

# ============================================================
# 4. Ablation: Gating Type
# ============================================================
gating = [
    {'Category': 'Ablation', 'Experiment': 'Gating Type', 'Method': 'Linear (unbounded)', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.7361, 'Std': '-', 'Notes': 'Best'},
    {'Category': 'Ablation', 'Experiment': 'Gating Type', 'Method': 'Linear (unbounded)', 'Dataset': 'Overall', 'Metric': 'AUC', 'Value': 0.6417, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Gating Type', 'Method': 'Sigmoid (bounded)', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.7083, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Gating Type', 'Method': 'Sigmoid (bounded)', 'Dataset': 'Overall', 'Metric': 'AUC', 'Value': 0.6254, 'Std': '-', 'Notes': '-'},
]

# ============================================================
# 5. Ablation: Backbone
# ============================================================
backbone = [
    {'Category': 'Ablation', 'Experiment': 'Backbone', 'Method': 'MLP (gated_mlp)', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.6278, 'Std': '-', 'Notes': 'Best'},
    {'Category': 'Ablation', 'Experiment': 'Backbone', 'Method': 'MLP (gated_mlp)', 'Dataset': 'Overall', 'Metric': 'AUC', 'Value': 0.6605, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Backbone', 'Method': 'Transformer', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.2479, 'Std': '-', 'Notes': 'Failed'},
    {'Category': 'Ablation', 'Experiment': 'Backbone', 'Method': 'Transformer', 'Dataset': 'Overall', 'Metric': 'AUC', 'Value': 0.5544, 'Std': '-', 'Notes': '-'},
]

# ============================================================
# 6. Ablation: Training Order
# ============================================================
training_order = [
    {'Category': 'Ablation', 'Experiment': 'Training Order', 'Method': 'select_then_mlp', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.6090, 'Std': '-', 'Notes': 'Best'},
    {'Category': 'Ablation', 'Experiment': 'Training Order', 'Method': 'select_then_mlp', 'Dataset': 'Overall', 'Metric': 'AUC', 'Value': 0.6628, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Training Order', 'Method': 'expand4', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.3694, 'Std': '-', 'Notes': 'Expansion hurts'},
    {'Category': 'Ablation', 'Experiment': 'Training Order', 'Method': 'expand8', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.4184, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Training Order', 'Method': 'expand16', 'Dataset': 'Overall', 'Metric': 'best-k', 'Value': 0.4181, 'Std': '-', 'Notes': '-'},
]

# ============================================================
# 7. Ablation: Polynomial Features
# ============================================================
polynomial = [
    {'Category': 'Ablation', 'Experiment': 'Polynomial', 'Method': 'degree=1, group', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Polynomial', 'Method': 'degree=2, expanded', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Polynomial', 'Method': 'degree=1, group', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.10, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Polynomial', 'Method': 'degree=2, group', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.40, 'Std': '-', 'Notes': '+30%'},
    {'Category': 'Ablation', 'Experiment': 'Polynomial', 'Method': 'degree=1, group', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 0.55, 'Std': '-', 'Notes': '-'},
    {'Category': 'Ablation', 'Experiment': 'Polynomial', 'Method': 'degree=2, group', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 0.75, 'Std': '-', 'Notes': 'Best: +20%'},
]

# ============================================================
# 8. NEW: Experiment A (Gradual ADMM + Pruning) - Updated with gradual pruning
# ============================================================
exp_a = [
    # Single-phase results (previous)
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_none', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': 'Baseline'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_soft', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': '-'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_hard', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': '-'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_none', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.33, 'Std': '0.24', 'Notes': 'Baseline'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_soft', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.83, 'Std': '0.24', 'Notes': 'Soft helps +50%'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_hard', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.33, 'Std': '0.24', 'Notes': 'Hard fails'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_none', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 0.25, 'Std': '0.00', 'Notes': 'Baseline'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_soft', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': 'Soft helps +75%'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Single-phase C=0.5', 'Method': 'gradual_hard', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 0.25, 'Std': '0.00', 'Notes': 'Hard fails'},
    # Gradual pruning results (NEW - better!)
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_none', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': 'Baseline'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_soft_no_rw', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': '-'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_hard_no_rw', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': 'Hard works!'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_none', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.33, 'Std': '0.24', 'Notes': 'Baseline'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_soft', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.17, 'Std': '0.24', 'Notes': 'RW hurts!'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_soft_no_rw', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': 'BEST! Soft+noRW'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_hard', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.17, 'Std': '0.24', 'Notes': 'RW hurts!'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_hard_no_rw', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.50, 'Std': '0.00', 'Notes': 'Gradual hard +50%'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_none', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 0.50, 'Std': '0.00', 'Notes': 'Baseline'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_soft', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': 'Soft+RW works'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_soft_no_rw', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.00', 'Notes': 'Soft+noRW works'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_hard', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 0.50, 'Std': '0.00', 'Notes': 'Baseline'},
    {'Category': 'NEW', 'Experiment': 'Exp A: Gradual 5-phase (delete 10%/phase)', 'Method': 'gradual_hard_no_rw', 'Dataset': 'Ring+XOR', 'Metric': 'best-k', 'Value': 0.50, 'Std': '0.00', 'Notes': '-'},
]

# ============================================================
# 9. Negative Results
# ============================================================
negative = [
    {'Category': 'Negative Results', 'Experiment': 'Transformer Pretrain', 'Method': 'MLP Baseline', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 1.00, 'Std': '0.0', 'Notes': 'Success rate 100%'},
    {'Category': 'Negative Results', 'Experiment': 'Transformer Pretrain', 'Method': 'Transformer + MAE Pretrain', 'Dataset': 'XOR', 'Metric': 'best-k', 'Value': 0.33, 'Std': '0.47', 'Notes': 'Success rate 33%'},
    {'Category': 'Negative Results', 'Experiment': 'Lottery Ticket', 'Method': 'single_pass', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.67, 'Std': '0.24', 'Notes': '-'},
    {'Category': 'Negative Results', 'Experiment': 'Lottery Ticket', 'Method': 'lottery_ticket (weight reset)', 'Dataset': 'Ring', 'Metric': 'best-k', 'Value': 0.08, 'Std': '0.19', 'Notes': 'FAILED!'},
]

# ============================================================
# 10. Hyperparameters
# ============================================================
hyperparams = [
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'Architecture', 'Dataset': '-', 'Metric': 'latent_size', 'Value': 32, 'Std': '-', 'Notes': '-'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'Architecture', 'Dataset': '-', 'Metric': 'n_hidden_layers', 'Value': 2, 'Std': '-', 'Notes': '-'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'Architecture', 'Dataset': '-', 'Metric': 'dropout', 'Value': 0.043, 'Std': '-', 'Notes': 'Tuned'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'Architecture', 'Dataset': '-', 'Metric': 'feat_drop', 'Value': 0.6, 'Std': '-', 'Notes': 'Tuned'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'Training', 'Dataset': '-', 'Metric': 'epochs', 'Value': 416, 'Std': '-', 'Notes': '100 warmup + 316 ADMM'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'Training', 'Dataset': '-', 'Metric': 'warmup_epochs', 'Value': 100, 'Std': '-', 'Notes': 'CRITICAL'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'Training', 'Dataset': '-', 'Metric': 'optimizer', 'Value': 'Adagrad', 'Std': '-', 'Notes': '-'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'Training', 'Dataset': '-', 'Metric': 'lr', 'Value': 0.00176, 'Std': '-', 'Notes': '-'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'ADMM', 'Dataset': '-', 'Metric': 'sparsity_penalty', 'Value': 'ratio_norm', 'Std': '-', 'Notes': 'L1/L2 ratio'},
    {'Category': 'Hyperparams', 'Experiment': 'SADMM-FS Default', 'Method': 'ADMM', 'Dataset': '-', 'Metric': 'C', 'Value': 0.05, 'Std': '-', 'Notes': 'Sparsity strength'},
]

# ============================================================
# 11. Dimension Scaling
# ============================================================
dim_scaling = [
    {'Category': 'Dimension Scaling', 'Experiment': 'XOR m scaling', 'Method': 'SADMM-FS', 'Dataset': 'XOR m=128', 'Metric': 'best-k', 'Value': 1.00, 'Std': '-', 'Notes': 'Perfect'},
    {'Category': 'Dimension Scaling', 'Experiment': 'XOR m scaling', 'Method': 'SADMM-FS', 'Dataset': 'XOR m=512', 'Metric': 'best-k', 'Value': 0.67, 'Std': '-', 'Notes': '-'},
    {'Category': 'Dimension Scaling', 'Experiment': 'XOR m scaling', 'Method': 'SADMM-FS', 'Dataset': 'XOR m=2048', 'Metric': 'best-k', 'Value': 0.17, 'Std': '-', 'Notes': 'Decays'},
    {'Category': 'Dimension Scaling', 'Experiment': 'XOR m scaling', 'Method': 'LassoNet', 'Dataset': 'XOR m=128', 'Metric': 'best-k', 'Value': 1.00, 'Std': '-', 'Notes': 'Perfect'},
    {'Category': 'Dimension Scaling', 'Experiment': 'XOR m scaling', 'Method': 'LassoNet', 'Dataset': 'XOR m=512', 'Metric': 'best-k', 'Value': 1.00, 'Std': '-', 'Notes': 'Best at m=512'},
    {'Category': 'Dimension Scaling', 'Experiment': 'XOR m scaling', 'Method': 'LassoNet', 'Dataset': 'XOR m=2048', 'Metric': 'best-k', 'Value': 0.17, 'Std': '-', 'Notes': '-'},
    {'Category': 'Dimension Scaling', 'Experiment': 'XOR m scaling', 'Method': 'TreeSHAP', 'Dataset': 'XOR (any m)', 'Metric': 'best-k', 'Value': 0.498, 'Std': '-', 'Notes': 'Constant'},
]

# Combine all
data = main_methods + realworld + iterative + gating + backbone + training_order + polynomial + exp_a + negative + hyperparams + dim_scaling

# Create DataFrame
df = pd.DataFrame(data)

# Save to xlsx
output_path = Path('E:/Projects/NEW_Pruning_20251110/custom_admm/results/ALL_METHODS_RESULTS_SUMMARY_v2.xlsx')
df.to_excel(output_path, index=False, sheet_name='Summary')

print(f'Saved to: {output_path}')
print(f'Total rows: {len(df)}')
print()
print('=' * 80)
print('KEY FINDINGS SUMMARY')
print('=' * 80)
print()
print('1. Main Methods:')
print('   - SADMM-FS: best-k 0.6278 (Best overall)')
print('   - STG: best-k 0.625')
print('   - TabNet/CAE/E2E-FS: best-k 0.17-0.29 (Failed)')
print()
print('2. Ablation Findings:')
print('   - Linear gate > Sigmoid gate (+3%)')
print('   - MLP > Transformer backbone (0.63 vs 0.25)')
print('   - Gradual ADMM helps Ring (+33%)')
print('   - Lottery Ticket hypothesis DOES NOT apply to FS')
print('   - Feature expansion hurts best-k')
print()
print('3. NEW Experiment A Results:')
print('   - Soft pruning helps Ring (+50%), Ring+XOR (+75%)')
print('   - Hard pruning FAILS (weight copy bug)')
print('   - Re-weighting doesnt hurt with small target_ratio=0.2')
print()
print('4. Real-World:')
print('   - SADMM-FS beats STG on all NIPS 2003 datasets')
print()
print('5. Negative Results:')
print('   - Transformer pretrain: 33% success vs 100% MLP')
print('   - Lottery Ticket: 0.08 vs 0.67 on Ring')
print()