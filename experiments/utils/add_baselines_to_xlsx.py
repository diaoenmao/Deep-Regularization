# -*- coding: utf-8 -*-
"""
Comprehensive baseline extraction script - fills all missing data in PAPER_RESULTS_TABLES.xlsx
Adds method type classification (Filter, Wrapper, Embedded, Intrinsic)
"""

import pandas as pd
from pathlib import Path
import json
import numpy as np

# Paths
BASELINE_DIR = Path('E:/Projects/NEW_Pruning_20251110/Feature-Selection-Benchmark/results')
EXTERNAL_DATA_DIR = BASELINE_DIR / 'external-data'

def extract_bestk_and_auc(method, dataset, m=128, n=1000):
    """Extract best-k and AUC values from baseline txt file."""
    filename = BASELINE_DIR / f"{method}-{dataset}-{n}.txt"
    if not filename.exists():
        return None, None

    with open(filename) as f:
        lines = f.readlines()

    if len(lines) == 0:
        return None, None

    header = lines[0].strip().split('\t')

    # Find columns
    bestk_col = auc_col = None
    for i, col in enumerate(header):
        if 'bestK' in col or 'best_k' in col.lower():
            bestk_col = i
        if 'AUC' in col:
            auc_col = i

    if bestk_col is None:
        bestk_col = 1
    if auc_col is None:
        auc_col = 3 if len(header) > 3 else None

    target_row = f"{dataset}_{m}_{n}"
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if parts[0] == target_row:
            try:
                bestk = float(parts[bestk_col])
                auc = float(parts[auc_col]) if auc_col is not None and len(parts) > auc_col else None
                return bestk, auc
            except:
                return None, None
    return None, None

def extract_realworld_auc(method, dataset):
    """Extract AUROC from external-data JSON file."""
    method_map = {
        'lassonet': 'lassonet', 'treeshap': 'treeshap', 'rf': 'rf',
        'mi': 'mi', 'mrmr': 'mrmr', 'relief': 'relief',
        'cae': 'cae', 'fsnet': 'fsnet', 'deeppink': 'deeppink',
        'canceloutsigmoid': 'canceloutsigmoid', 'canceloutsoftmax': 'canceloutsoftmax',
        'nn': 'nn',
    }

    bench_method = method_map.get(method, method.lower())
    filename = EXTERNAL_DATA_DIR / f"{dataset}-{bench_method}.json"

    if not filename.exists():
        return None

    try:
        with open(filename) as f:
            data = json.load(f)
        return data.get('auroc', None)
    except:
        return None

# Method type classification based on Nature Scientific Reports paper and domain knowledge
# Filter: statistical measures, no model training
# Wrapper: uses model to evaluate feature subsets
# Embedded: feature selection integrated into model training
# Intrinsic: uses model's internal weights/importance
# Deep Learning: neural network based methods

# Method type classification based on Nature Scientific Reports paper (s41598-024-82583-5)
# Paper divides methods into two groups:
# 1. Instance-level feature attribution (post-hoc)
# 2. Embedded/Filter FS methods

METHOD_TYPES = {
    # Deep Learning (Embedded) - feature selection integrated into training
    'SADMM-FS': 'Embedded (Deep Learning)',
    'STG': 'Embedded (Deep Learning)',
    'TabNet': 'Embedded (Deep Learning)',
    'CAE': 'Embedded (Deep Learning)',
    'cae': 'Embedded (Deep Learning)',
    'FSNet': 'Embedded (Deep Learning)',
    'fsnet': 'Embedded (Deep Learning)',
    'DeepPINK': 'Embedded (Deep Learning)',
    'deeppink': 'Embedded (Deep Learning)',
    'E2E-FS': 'Embedded (Deep Learning)',
    'CancelOut': 'Embedded (Deep Learning)',
    'canceloutsigmoid': 'Embedded (Deep Learning)',
    'canceloutsoftmax': 'Embedded (Deep Learning)',
    'lassonet': 'Embedded (Deep Learning)',
    'LassoNet': 'Embedded (Deep Learning)',

    # Tree-based (Embedded) - feature importance from tree structure
    'RF': 'Embedded (Tree-based)',
    'TreeSHAP': 'Embedded (Tree-based)',
    'rf': 'Embedded (Tree-based)',
    'treeshap': 'Embedded (Tree-based)',

    # Filter methods - statistical measures, no model training
    'mi': 'Filter',
    'MI': 'Filter',
    'mrmr': 'Filter',
    'MRMR': 'Filter',
    'relief': 'Filter',
    'Relief': 'Filter',

    # Instance-level feature attribution (post-hoc) - analyze trained model
    # All SA_METHODS from benchmark (10 methods + nn)
    'nn': 'Attribution (Post-hoc)',
    'NN': 'Attribution (Post-hoc)',
    'nnfs': 'Attribution (Post-hoc)',  # Saliency method in benchmark
    'Saliency': 'Attribution (Post-hoc)',
    'InputXGradient': 'Attribution (Post-hoc)',
    'IG_noMul': 'Attribution (Post-hoc)',  # Integrated Gradient
    'Integrated gradient': 'Attribution (Post-hoc)',
    'SmoothGrad': 'Attribution (Post-hoc)',
    'GuidedBackprop': 'Attribution (Post-hoc)',
    'Guided backpropagation': 'Attribution (Post-hoc)',
    'DeepLift': 'Attribution (Post-hoc)',
    'Deconvolution': 'Attribution (Post-hoc)',
    'FeatureAblation': 'Attribution (Post-hoc)',
    'Feature ablation': 'Attribution (Post-hoc)',
    'FeaturePermutation': 'Attribution (Post-hoc)',
    'Feature permutation': 'Attribution (Post-hoc)',
    'ShapleyValueSampling': 'Attribution (Post-hoc)',
    'Shapley value sampling': 'Attribution (Post-hoc)',
}

# Publication year for sorting (chronological order within each type)
# Filter methods (earliest)
# Relief: 1994 (Kononenko), mRMR: 2005 (Peng), MI: classic (1950s, use 1960 as placeholder)
METHOD_YEAR = {
    # Filter methods (chronological)
    'relief': 1994, 'Relief': 1994,  # ReliefF - Kononenko 1994
    'mrmr': 2005, 'MRMR': 2005,      # mRMR - Peng et al. 2005
    'mi': 1960, 'MI': 1960,          # Mutual Information - Shannon 1948, but use 1960 for FS context

    # Embedded (Tree-based)
    'rf': 2001, 'RF': 2001,          # Random Forest - Breiman 2001
    'treeshap': 2020, 'TreeSHAP': 2020,  # TreeSHAP - Lundberg et al. 2020

    # Embedded (Deep Learning) - chronological by publication
    'E2E-FS': 2018,                        # E2E-FS - He et al. 2018 (End-to-End FS)
    'DeepPINK': 2018, 'deeppink': 2018,     # DeepPINK - Chen et al. 2018
    'FSNet': 2020, 'fsnet': 2020,          # FSNet - 2020
    'TabNet': 2020,                        # TabNet - Arik & Pfister 2020
    'CAE': 2020, 'cae': 2020,              # Concrete Autoencoder - Abid et al. 2020
    'canceloutsigmoid': 2020, 'canceloutsoftmax': 2020,  # CancelOut - Avram et al. 2020
    'CancelOut': 2020,                     # CancelOut (自有实验) - 2020
    'lassonet': 2020, 'LassoNet': 2020,    # LassoNet - Lemhadri et al. 2020
    'STG': 2020,                           # STG - Yamada et al. 2020
    'SADMM-FS': 2026,                      # Our method - 2026

    # Attribution (Post-hoc) - chronological by publication
    'nn': 2013, 'NN': 2013,               # Saliency - Simonyan et al. 2013 (VGG paper)
    'nnfs': 2013, 'Saliency': 2013,       # Saliency - same origin
    'Deconvolution': 2014,                # DeconvNet - Zeiler & Fergus 2014
    'GuidedBackprop': 2014, 'Guided backpropagation': 2014,  # Springenberg et al. 2014
    'DeepLift': 2017,                     # DeepLIFT - Shrikumar et al. 2017
    'Integrated gradient': 2017, 'IG_noMul': 2017,  # Sundararajan et al. 2017
    'SmoothGrad': 2017,                   # Smilkov et al. 2017
    'InputXGradient': 2017, 'Input x Gradient': 2017,  # Approximation to IG
    'FeatureAblation': 2018, 'Feature ablation': 2018,  # General ablation
    'FeaturePermutation': 2018, 'Feature permutation': 2018,  # General permutation
    'ShapleyValueSampling': 2019, 'Shapley value sampling': 2019,  # SHAP - Lundberg & Lee 2017, sampling variant 2019
}

# Sort order for method types (groups)
TYPE_ORDER = {
    'Filter': 1,
    'Embedded (Tree-based)': 2,
    'Embedded (Deep Learning)': 3,
    'Attribution (Post-hoc)': 4,
}

# Method full names for display
METHOD_NAMES = {
    'mi': 'Mutual Information',
    'mrmr': 'mRMR',
    'relief': 'ReliefF',
    'rf': 'Random Forest',
    'treeshap': 'TreeSHAP',
    'lassonet': 'LassoNet',
    'nn': 'Neural Network',
    'cae': 'CAE',
    'fsnet': 'FSNet',
    'deeppink': 'DeepPINK',
    'canceloutsigmoid': 'CancelOut (Sigmoid)',
    'canceloutsoftmax': 'CancelOut (Softmax)',
    'e2efs': 'E2E-FS',
    # SA_METHODS display names
    'nnfs': 'Saliency',
    'Saliency': 'Saliency',
    'InputXGradient': 'Input × Gradient',
    'IG_noMul': 'Integrated Gradients',
    'SmoothGrad': 'SmoothGrad',
    'GuidedBackprop': 'Guided Backprop',
    'DeepLift': 'DeepLIFT',
    'Deconvolution': 'Deconvolution',
    'FeatureAblation': 'Feature Ablation',
    'FeaturePermutation': 'Feature Permutation',
    'ShapleyValueSampling': 'Shapley Value Sampling',
}

# Synthetic datasets
datasets_map = {
    'XOR (k=2)': ('xor', 'xor'),
    'Ring (k=2)': ('ring', 'ring'),
    'Ring+XOR (k=4)': ('ring+xor', 'ring+xor'),
    'Ring+XOR+Sum (k=4)': ('ring+xor+sum', 'dag'),
}

# Real-world datasets
realworld_datasets = ['madelon', 'gisette', 'arcene', 'dexter']
dataset_m_map = {'madelon': 500, 'gisette': 5000, 'arcene': 10000, 'dexter': 20000}

# Method mappings
method_benchmark_map = {
    'CAE': 'cae', 'FSNet': 'fsnet', 'DeepPINK': 'deeppink',
    'CancelOut': 'canceloutsigmoid',
    'TreeSHAP': 'treeshap', 'RF': 'rf',
    'lassonet': 'lassonet', 'mi': 'mi', 'mrmr': 'mrmr',
    'relief': 'relief', 'nn': 'nn',
    'canceloutsigmoid': 'canceloutsigmoid', 'canceloutsoftmax': 'canceloutsoftmax',
    'treeshap': 'treeshap', 'rf': 'rf',
    'E2E-FS': 'e2efs', 'e2efs': 'e2efs',
    # SA_METHODS mappings
    'nnfs': 'nnfs', 'Saliency': 'Saliency',
    'InputXGradient': 'InputXGradient', 'IG_noMul': 'IG_noMul',
    'SmoothGrad': 'SmoothGrad', 'GuidedBackprop': 'GuidedBackprop',
    'DeepLift': 'DeepLift', 'Deconvolution': 'Deconvolution',
    'FeatureAblation': 'FeatureAblation', 'FeaturePermutation': 'FeaturePermutation',
    'ShapleyValueSampling': 'ShapleyValueSampling',
}

# Pure baseline methods (from benchmark)
# Note: E2E-FS is NOT in benchmark results, keep 自有实验 data
# SA_METHODS (Saliency Attribution) - all have benchmark data
pure_baseline_methods = [
    'lassonet', 'mi', 'mrmr', 'relief', 'nn',
    'canceloutsigmoid', 'canceloutsoftmax',
    'treeshap', 'rf', 'cae', 'fsnet', 'deeppink',
    # SA_METHODS (Attribution methods)
    'nnfs', 'Saliency', 'InputXGradient', 'IG_noMul', 'SmoothGrad',
    'GuidedBackprop', 'DeepLift', 'Deconvolution',
    'FeatureAblation', 'FeaturePermutation', 'ShapleyValueSampling'
]

# Methods to REMOVE (自有实验，不使用benchmark数据)
# 这些方法的Match Level是'-'或'full_match'，数据来源不明，应该用benchmark数据替代
methods_to_remove = ['TreeSHAP', 'RF', 'DeepPINK', 'CancelOut']  # 大写的自有实验版本 + CancelOut自有实验

# Read xlsx
xlsx_path = Path('E:/Projects/NEW_Pruning_20251110/results/tables/PAPER_RESULTS_TABLES.xlsx')
existing_sheets = pd.read_excel(xlsx_path, sheet_name=None)
main_synthetic = existing_sheets['Table1_Main_Synthetic']
realworld_table = existing_sheets['Table2_Main_RealWorld']

print("=== Processing Table1_Main_Synthetic ===")

# Add Method Type column if not exists
if 'Method Type' not in main_synthetic.columns:
    main_synthetic.insert(1, 'Method Type', '')

# 1. Fill Method Type for existing rows
for idx, row in main_synthetic.iterrows():
    method = row['Method']
    method_type = METHOD_TYPES.get(method, METHOD_TYPES.get(method.lower(), '-'))
    main_synthetic.at[idx, 'Method Type'] = method_type

# 2. Fill missing values and compute Mean best-k
for idx, row in main_synthetic.iterrows():
    method = row['Method']
    bench_method = method_benchmark_map.get(method, method.lower())

    bestk_values = []
    auc_values = []

    for col_name, (ds_file1, ds_file2) in datasets_map.items():
        current_val = row[col_name]

        if current_val == '-' or pd.isna(current_val):
            bestk, auc = extract_bestk_and_auc(bench_method, ds_file1, m=128)
            if bestk is None:
                bestk, auc = extract_bestk_and_auc(bench_method, ds_file2, m=128)
            if bestk is not None:
                main_synthetic.at[idx, col_name] = bestk
                bestk_values.append(bestk)
                if auc is not None:
                    auc_values.append(auc)
        else:
            val = row[col_name]
            if val != '-' and not pd.isna(val):
                bestk_values.append(float(val))

    # Compute Mean best-k
    valid_bestk = [v for v in bestk_values if v is not None and v != '-']
    if valid_bestk:
        mean_bestk = np.mean(valid_bestk)
        if row['Mean best-k'] == '-' or pd.isna(row['Mean best-k']):
            main_synthetic.at[idx, 'Mean best-k'] = round(mean_bestk, 4)

    # Compute Mean AUC (only for non-Filter methods that have AUC)
    method_type = METHOD_TYPES.get(method, METHOD_TYPES.get(method.lower(), ''))
    if 'Filter' not in method_type:  # Filter methods don't have AUC
        if auc_values and (row['Mean AUC'] == '-' or pd.isna(row['Mean AUC'])):
            valid_auc = [v for v in auc_values if v is not None]
            if valid_auc:
                main_synthetic.at[idx, 'Mean AUC'] = round(np.mean(valid_auc), 4)

# 3. Remove duplicate benchmark rows and自有实验行
rows_to_keep = []
for idx, row in main_synthetic.iterrows():
    method = row['Method']
    match_level = row.get('Match Level', '')

    # Remove自有实验的TreeSHAP/RF (Match Level='-'，数据来源不明)
    if method in methods_to_remove:
        continue

    # Remove duplicate 'benchmark' rows
    if match_level == 'benchmark':
        continue

    # Remove old pure baseline rows (we'll add fresh ones from benchmark)
    if method.lower() in pure_baseline_methods and match_level == 'standard':
        continue

    rows_to_keep.append(idx)

main_synthetic = main_synthetic.loc[rows_to_keep].reset_index(drop=True)

# 4. Add pure baseline methods with full data
for method in pure_baseline_methods:
    bench_method = method_benchmark_map.get(method, method.lower())
    method_type = METHOD_TYPES.get(method, METHOD_TYPES.get(bench_method, '-'))

    row_data = {
        'Method': method,
        'Method Type': method_type,
        'Match Level': 'standard'
    }
    bestk_values = []
    auc_values = []

    for col_name, (ds_file1, ds_file2) in datasets_map.items():
        bestk, auc = extract_bestk_and_auc(bench_method, ds_file1, m=128)
        if bestk is None:
            bestk, auc = extract_bestk_and_auc(bench_method, ds_file2, m=128)
        row_data[col_name] = bestk if bestk is not None else '-'
        if bestk is not None:
            bestk_values.append(bestk)
        if auc is not None:
            auc_values.append(auc)

    if not bestk_values:
        continue

    row_data['Mean best-k'] = round(np.mean(bestk_values), 4)

    # AUC only for non-Filter methods
    if 'Filter' not in method_type:
        row_data['Mean AUC'] = round(np.mean(auc_values), 4) if auc_values else 'N/A'
    else:
        row_data['Mean AUC'] = 'N/A (Filter)'  # Explicitly note why no AUC

    main_synthetic = pd.concat([main_synthetic, pd.DataFrame([row_data])], ignore_index=True)
    print(f"  {method}: {method_type}, mean_best-k={row_data['Mean best-k']}, AUC={row_data['Mean AUC']}")

# Reorder columns
cols = ['Method', 'Method Type', 'Match Level', 'XOR (k=2)', 'Ring (k=2)', 'Ring+XOR (k=4)', 'Ring+XOR+Sum (k=4)', 'Mean best-k', 'Mean AUC']
main_synthetic = main_synthetic[[c for c in cols if c in main_synthetic.columns]]

# Sort by Method Type (group order), then by Year (chronological within type)
def get_sort_key(row):
    method = row['Method']
    method_type = METHOD_TYPES.get(method, METHOD_TYPES.get(method.lower(), 'Other'))
    year = METHOD_YEAR.get(method, METHOD_YEAR.get(method.lower(), 9999))  # Unknown methods go to end
    type_order = TYPE_ORDER.get(method_type, 99)  # Unknown types go to end
    return (type_order, year)

main_synthetic['_sort_key'] = main_synthetic.apply(get_sort_key, axis=1)
main_synthetic = main_synthetic.sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)

print("\n=== Processing Table2_Main_RealWorld ===")

# Remove existing baseline rows and自有实验RF from realworld_table
realworld_baseline_methods = ['lassonet', 'treeshap', 'rf', 'mi', 'mrmr', 'relief', 'cae', 'fsnet', 'deeppink', 'canceloutsigmoid', 'RF']  # Add RF (自有实验)
rows_to_keep = []
for idx, row in realworld_table.iterrows():
    method = row['Method']
    # Remove both lowercase baseline and uppercase自有实验RF
    if method.lower() in realworld_baseline_methods or method in methods_to_remove:
        continue
    rows_to_keep.append(idx)
realworld_table = realworld_table.loc[rows_to_keep].reset_index(drop=True)

# Add Method Type column to realworld table
if 'Method Type' not in realworld_table.columns:
    realworld_table.insert(1, 'Method Type', '')

# Fill Method Type for existing rows
for idx, row in realworld_table.iterrows():
    method = row['Method']
    method_type = METHOD_TYPES.get(method, METHOD_TYPES.get(method.lower(), '-'))
    realworld_table.at[idx, 'Method Type'] = method_type

# Add baseline methods
for method in ['lassonet', 'treeshap', 'rf', 'mi', 'mrmr', 'relief', 'cae', 'fsnet', 'deeppink', 'canceloutsigmoid']:
    method_type = METHOD_TYPES.get(method, METHOD_TYPES.get(method.lower(), '-'))

    row_data = {'Method': method, 'Method Type': method_type}
    auc_values = []

    for dataset in realworld_datasets:
        auc = extract_realworld_auc(method, dataset)
        col_name = f"{dataset} (m={dataset_m_map[dataset]})"
        row_data[col_name] = auc if auc is not None else '-'
        if auc is not None:
            auc_values.append(auc)

    if auc_values:
        row_data['Mean AUROC'] = round(np.mean(auc_values), 4)
        realworld_table = pd.concat([realworld_table, pd.DataFrame([row_data])], ignore_index=True)
        print(f"  {method}: {method_type}, mean_AUROC={row_data['Mean AUROC']}")

# Sort realworld table by Method Type, then by Year
realworld_table['_sort_key'] = realworld_table.apply(get_sort_key, axis=1)
realworld_table = realworld_table.sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)

print("\n=== Final Results ===")
print("\nTable1_Main_Synthetic:")
print(main_synthetic.to_string())
print("\nTable2_Main_RealWorld:")
print(realworld_table.to_string())

# Save
with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
    main_synthetic.to_excel(writer, sheet_name='Table1_Main_Synthetic', index=False)
    realworld_table.to_excel(writer, sheet_name='Table2_Main_RealWorld', index=False)
    for sheet_name, df in existing_sheets.items():
        if sheet_name not in ['Table1_Main_Synthetic', 'Table2_Main_RealWorld']:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\n=== Updated: {xlsx_path} ===")