#!/usr/bin/env python
"""Aggregate existing Feature-Selection-Benchmark results (compact set) and
produce a CSV summary plus a comparison PNG.

Usage:
    python scripts/aggregate_fs_results.py

Outputs:
    results/compare_methods_compact-<date>.csv
    results/figures/compare_methods_compact-<date>.png
"""
import os
import re
import csv
import datetime
from collections import defaultdict, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS_RESULTS = os.path.join(ROOT, 'Feature-Selection-Benchmark', 'results')
OUT_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

TARGET_DATASETS = ['xor', 'ring', 'ring+xor', 'ring+xor+sum']
TARGET_NS = [8, 32, 128]

# Helper to parse a data row like: "ring+xor+sum_8_500\t0.8055\t1.0\t..."
def parse_row(first_field, parts):
    # split last two underscores for n_features and sample
    try:
        dataset, n_str, samp_str = first_field.rsplit('_', 2)
    except ValueError:
        return None
    try:
        n = int(n_str)
    except ValueError:
        return None
    # best_k, best_2k expected at parts[1], parts[2]
    try:
        best_k = float(parts[1])
    except Exception:
        best_k = None
    try:
        best_2k = float(parts[2])
    except Exception:
        best_2k = None
    return dataset, n, best_k, best_2k

# Scan files
files = [f for f in os.listdir(FS_RESULTS) if f.endswith('.txt')]
methods = set()
results = defaultdict(dict)  # (method)->(dataset,n)->(best_k,best_2k)

for fname in files:
    method = fname.split('-', 1)[0]
    methods.add(method)
    path = os.path.join(FS_RESULTS, fname)
    with open(path, 'r', encoding='utf-8') as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) < 2:
        continue
    header = lines[0]
    for ln in lines[1:]:
        parts = re.split(r'\s+', ln)
        parsed = parse_row(parts[0], parts)
        if parsed is None:
            continue
        dataset, n, best_k, best_2k = parsed
        if dataset not in TARGET_DATASETS:
            continue
        if n not in TARGET_NS:
            continue
        results[method][(dataset, n)] = (best_k, best_2k)

methods = sorted(methods)
# Build CSV
date = datetime.datetime.now().strftime('%Y%m%d')
csv_path = os.path.join(OUT_DIR, f'compare_methods_compact-{date}.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
    w = csv.writer(csvf)
    header = ['method', 'dataset', 'n', 'best_k', 'best_2k']
    w.writerow(header)
    for method in methods:
        for dataset in TARGET_DATASETS:
            for n in TARGET_NS:
                key = (dataset, n)
                if key in results[method]:
                    bk, b2k = results[method][key]
                else:
                    bk, b2k = '', ''
                w.writerow([method, dataset, n, bk, b2k])
print(f'[Saved] CSV summary: {csv_path}')

# Create a simple comparison plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Filter methods that have at least one entry for n=128 in our target datasets
sel_methods = [m for m in methods if any(((d,128) in results[m]) for d in TARGET_DATASETS)]
if not sel_methods:
    print('No methods found with target data -> skipping plot')
else:
    n_methods = len(sel_methods)
    fig, axes = plt.subplots(2, 2, figsize=(max(6, 1.5 * n_methods), 8), squeeze=False)
    axes = axes.flatten()
    colors = plt.get_cmap('tab20').colors
    for i, dataset in enumerate(TARGET_DATASETS):
        ax = axes[i]
        x = np.arange(len(sel_methods))
        width = 0.25
        for j, n in enumerate(TARGET_NS):
            vals = []
            for m in sel_methods:
                val = results[m].get((dataset, n), (None, None))[0]
                vals.append(val if val is not None else 0.0)
            ax.bar(x + (j - 1) * width, vals, width, label=f'n={n}', color=colors[j % len(colors)])
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(sel_methods, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('best-k')
        ax.legend()
    plt.tight_layout()
    png_path = os.path.join(FIG_DIR, f'compare_methods_compact-{date}.png')
    plt.savefig(png_path, dpi=150)
    print(f'[Saved] Plot: {png_path}')

print('Done.')
