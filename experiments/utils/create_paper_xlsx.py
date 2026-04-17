# -*- coding: utf-8 -*-
"""
Create paper-ready xlsx with organized sheets:
- Table 1: Main Results (Synthetic Benchmark)
- Table 2: Main Results (Real-World)
- Table 3: Ablation Studies
- Table 4: New Experiment A (Pruning Comparison)
- Table 5: Negative Results
- Appendix: Hyperparameters
"""

import pandas as pd
from pathlib import Path

# ============================================================
# Table 1: Main Results (Synthetic Benchmark)
# ============================================================
main_synthetic = pd.DataFrame({
    'Method': ['SADMM-FS', 'STG', 'CancelOut', 'TabNet', 'CAE', 'E2E-FS', 'FSNet', 'DeepPINK', 'TreeSHAP', 'RF'],
    'Match Level': ['method_specific', 'backbone_only', 'full_match', 'standard', 'standard', 'standard', 'standard', 'full_match', '-', '-'],
    'XOR (k=2)': [1.00, 1.00, '-', '-', '-', '-', '-', '-', 0.498, 0.493],
    'Ring (k=2)': [0.50, 0.00, '-', '-', '-', '-', '-', '-', 0.990, 1.00],
    'Ring+XOR (k=4)': [0.625, 0.500, '-', '-', '-', '-', '-', '-', 0.400, 0.350],
    'Ring+XOR+Sum (k=4)': [0.667, 0.500, '-', '-', '-', '-', '-', '-', 0.500, 0.600],
    'Mean best-k': [0.6278, 0.625, 0.625, 0.2851, 0.1861, 0.1767, 0.1667, 0.0938, '-', '-'],
    'Mean AUC': [0.6605, '-', '-', 0.5995, 0.5236, 0.5117, 0.5621, '-', '-', '-'],
})

# ============================================================
# Table 2: Main Results (Real-World)
# ============================================================
main_realworld = pd.DataFrame({
    'Method': ['SADMM-FS', 'STG', 'RF'],
    'madelon (m=500)': [0.965, 0.847, 0.965],
    'gisette (m=5000)': [0.985, 0.963, 0.975],
    'arcene (m=10000)': [0.887, 0.808, 0.846],
    'dexter (m=20000)': [0.889, 0.825, 0.779],
    'Mean AUROC': [0.932, 0.861, 0.891],
})

# ============================================================
# Table 3: Ablation Studies
# ============================================================

# 3a: Gating Type
ablation_gating = pd.DataFrame({
    'Gate Type': ['Linear (unbounded)', 'Sigmoid (bounded)'],
    'XOR best-k': [1.00, 1.00],
    'Ring best-k': [0.67, 0.58],
    'Ring+XOR best-k': [0.54, 0.54],
    'Mean best-k': [0.7361, 0.7083],
    'Mean AUC': [0.6417, 0.6254],
})

# 3b: Backbone
ablation_backbone = pd.DataFrame({
    'Backbone': ['MLP (gated_mlp)', 'Transformer'],
    'XOR best-k': [1.00, '-'],
    'Ring best-k': [0.50, '-'],
    'Ring+XOR best-k': [0.625, '-'],
    'Mean best-k': [0.6278, 0.2479],
    'Mean AUC': [0.6605, 0.5544],
})

# 3c: Iterative FS Strategy
ablation_iterative = pd.DataFrame({
    'Strategy': ['single_pass', 'iterative_hard', 'lottery_ticket', 'gradual_admm'],
    'XOR best-k': [1.00, 0.67, 0.50, 1.00],
    'Ring best-k': [0.67, 0.58, 0.08, 1.00],
    'Ring+XOR best-k': [0.67, 0.13, 0.13, 0.67],
    'Notes': ['Baseline', 'Hard prune hurts', 'FAILED', 'Best on Ring'],
})

# 3d: Training Order
ablation_training = pd.DataFrame({
    'Order': ['select_then_mlp', 'expand4', 'expand8', 'expand16'],
    'Mean best-k': [0.6090, 0.3694, 0.4184, 0.4181],
    'Mean AUC': [0.6628, 0.6221, 0.6383, 0.6537],
    'Notes': ['Best', 'Expansion hurts', '-', '-'],
})

# 3e: Polynomial Features
ablation_polynomial = pd.DataFrame({
    'Config': ['degree=1, group', 'degree=2, expanded', 'degree=2, group'],
    'XOR best-k': [1.00, 1.00, '-'],
    'Ring best-k': [0.10, '-', 0.40],
    'Ring+XOR best-k': [0.55, '-', 0.75],
    'Notes': ['Baseline', '-', 'Best on Ring+XOR +20%'],
})

# ============================================================
# Table 4: NEW Experiment A (Pruning Comparison)
# ============================================================

# 4a: Single-phase (C=0.5)
exp_single = pd.DataFrame({
    'Variant': ['none (baseline)', 'soft (mask)', 'soft_no_rw', 'hard (delete)', 'hard_no_rw'],
    'XOR best-k': [1.00, 1.00, 1.00, 1.00, 1.00],
    'Ring best-k': [0.33, 0.83, 0.83, 0.33, 0.33],
    'Ring+XOR best-k': [0.25, 1.00, 1.00, 0.25, 0.25],
    'Notes': ['Baseline', 'Soft helps +50%', 'Same', 'Hard fails', 'Hard fails'],
})

# 4b: Gradual 5-phase (delete 10%/phase)
exp_gradual = pd.DataFrame({
    'Variant': ['none (baseline)', 'soft (mask+rw)', 'soft_no_rw (mask)', 'hard (delete+rw)', 'hard_no_rw (delete)'],
    'XOR best-k': [1.00, 1.00, 1.00, 1.00, 1.00],
    'Ring best-k': [0.33, 0.17, 1.00, 0.17, 0.50],
    'Ring+XOR best-k': [0.50, 1.00, 1.00, 0.50, 0.50],
    'Notes': ['Baseline', 'RW hurts!', 'BEST! Perfect on Ring', 'RW hurts!', 'Gradual works +50%'],
})

# ============================================================
# Table 5: Negative Results
# ============================================================
negative_results = pd.DataFrame({
    'Experiment': ['Transformer Pretrain', 'Transformer Pretrain', 'Lottery Ticket', 'Lottery Ticket'],
    'Method': ['MLP Baseline', 'Transformer + MAE Pretrain', 'single_pass', 'lottery_ticket (reset)'],
    'Dataset': ['XOR', 'XOR', 'Ring', 'Ring'],
    'best-k': [1.00, 0.33, 0.67, 0.08],
    'Std': ['0.0', '0.47', '0.24', '0.19'],
    'Success Rate': ['100%', '33%', '-', '-'],
    'Root Cause': ['-', 'No spatial structure in tabular data', '-', 'Weight reset breaks learned features'],
})

# ============================================================
# Appendix: Hyperparameters
# ============================================================
hyperparams = pd.DataFrame({
    'Component': ['Architecture', 'Architecture', 'Architecture', 'Architecture', 'Architecture',
                  'Training', 'Training', 'Training', 'Training', 'Training',
                  'ADMM', 'ADMM', 'ADMM'],
    'Parameter': ['latent_size', 'n_hidden_layers', 'dropout', 'feat_drop', 'activation',
                  'epochs', 'warmup_epochs', 'optimizer', 'lr', 'batch_size',
                  'C', 'rho', 'penalty'],
    'Value': [32, 2, 0.043, 0.6, 'mish',
              416, 100, 'Adagrad', 0.00176, 56,
              0.05, 'adaptive', 'ratio_norm'],
    'Notes': ['-', '-', 'Tuned', 'Tuned', '-',
              '100 warmup + 316 ADMM', 'CRITICAL', '-', '-', '-',
              'Sparsity strength', 'Boyd §3.4.1', 'L1/L2 ratio'],
})

# ============================================================
# Save to xlsx with multiple sheets
# ============================================================
output_path = Path('E:/Projects/NEW_Pruning_20251110/custom_admm/results/PAPER_RESULTS_TABLES.xlsx')

with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    main_synthetic.to_excel(writer, sheet_name='Table1_Main_Synthetic', index=False)
    main_realworld.to_excel(writer, sheet_name='Table2_Main_RealWorld', index=False)

    # Ablation tables
    ablation_gating.to_excel(writer, sheet_name='Table3a_Ablation_Gating', index=False)
    ablation_backbone.to_excel(writer, sheet_name='Table3b_Ablation_Backbone', index=False)
    ablation_iterative.to_excel(writer, sheet_name='Table3c_Ablation_Iterative', index=False)
    ablation_training.to_excel(writer, sheet_name='Table3d_Ablation_TrainingOrder', index=False)
    ablation_polynomial.to_excel(writer, sheet_name='Table3e_Ablation_Polynomial', index=False)

    # Experiment A
    exp_single.to_excel(writer, sheet_name='Table4a_ExpA_SinglePhase', index=False)
    exp_gradual.to_excel(writer, sheet_name='Table4b_ExpA_GradualPhase', index=False)

    # Negative results
    negative_results.to_excel(writer, sheet_name='Table5_NegativeResults', index=False)

    # Appendix
    hyperparams.to_excel(writer, sheet_name='Appendix_Hyperparams', index=False)

print(f'Saved to: {output_path}')
print()
print('Sheets created:')
print('  Table1_Main_Synthetic - Main methods comparison on synthetic')
print('  Table2_Main_RealWorld - Real-world dataset results')
print('  Table3a-3e - Ablation studies (Gating, Backbone, Iterative, Training, Polynomial)')
print('  Table4a-4b - Experiment A (Single-phase vs Gradual)')
print('  Table5_NegativeResults - Negative findings')
print('  Appendix_Hyperparams - Default configuration')
print()
print('=' * 80)
print('KEY INSIGHT: gradual_soft_no_rw achieves PERFECT 1.00 on Ring!')
print('=' * 80)