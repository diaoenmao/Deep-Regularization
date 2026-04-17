#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare admm_input_group results with Feature-Selection-Benchmark methods.
"""
import pandas as pd
import numpy as np

# Our admm_input_group results (Mar 19 run)
admm_xor = [1.0, 0.75, 0.1667, 0.0833, 0.0, 0.0, 0.3333, 0.3333, 0.8333, 0.4167, 0.4167]
admm_ring = [0.3333, 0.0, 0.0, 0.4167, 0.0, 0.1667, 0.0, 0.1667, 0.0833]
admm_rx = [1.0, 0.5, 0.2083, 0.125, 0.3333, 0.375, 0.3333, 0.1667, 0.0, 0.0]
admm_rxsum = [1.0, 0.8056, 0.4444, 0.4722, 0.5278, 0.6111, 0.5278, 0.4444, 0.4444, 0.3333]

admm_avg = {
    'XOR': np.mean(admm_xor),
    'RING': np.mean(admm_ring),
    'RING+XOR': np.mean(admm_rx),
    'RING+XOR+SUM': np.mean(admm_rxsum),
}

# Benchmark results from table-1000.tex (N=1000)
# Averaged over m ∈ {2,4,8,16,32,64,128,256,512,1024,2048}
benchmark_methods = {
    'Saliency': {'RING': 31.8, 'XOR': 57.6, 'RING+XOR': 34.6, 'RING+XOR+SUM': 53.9},
    'Input×Gradient': {'RING': 34.1, 'XOR': 57.6, 'RING+XOR': 34.2, 'RING+XOR+SUM': 54.7},
    'DeepLift': {'RING': 33.3, 'XOR': 56.8, 'RING+XOR': 34.6, 'RING+XOR+SUM': 53.9},
    'SmoothGrad': {'RING': 31.8, 'XOR': 57.6, 'RING+XOR': 34.2, 'RING+XOR+SUM': 54.2},
    'Feature Ablation': {'RING': 29.5, 'XOR': 57.6, 'RING+XOR': 34.2, 'RING+XOR+SUM': 53.3},
    'mRMR': {'RING': 100.0, 'XOR': 11.4, 'RING+XOR': 81.7, 'RING+XOR+SUM': 74.7},
    'LassoNet': {'RING': 34.8, 'XOR': 81.8, 'RING+XOR': 44.6, 'RING+XOR+SUM': 64.2},
    'Relief': {'RING': 40.2, 'XOR': 72.7, 'RING+XOR': 37.1, 'RING+XOR+SUM': 43.9},
    'Concrete Autoencoder': {'RING': 19.7, 'XOR': 22.7, 'RING+XOR': 19.2, 'RING+XOR+SUM': 25.8},
    'FSNet': {'RING': 20.5, 'XOR': 18.2, 'RING+XOR': 21.2, 'RING+XOR+SUM': 25.3},
    'CancelOut (sigmoid)': {'RING': 34.1, 'XOR': 60.6, 'RING+XOR': 37.5, 'RING+XOR+SUM': 55.8},
    'DeepPINK': {'RING': 21.2, 'XOR': 34.1, 'RING+XOR': 21.2, 'RING+XOR+SUM': 48.3},
    'Random Forest': {'RING': 100.0, 'XOR': 54.5, 'RING+XOR': 88.8, 'RING+XOR+SUM': 85.3},
    'TreeSHAP': {'RING': 100.0, 'XOR': 40.2, 'RING+XOR': 81.7, 'RING+XOR+SUM': 78.3},
}

# Add our method
benchmark_methods['ADMM Input Group (Ours)'] = {
    'RING': admm_avg['RING'] * 100,
    'XOR': admm_avg['XOR'] * 100,
    'RING+XOR': admm_avg['RING+XOR'] * 100,
    'RING+XOR+SUM': admm_avg['RING+XOR+SUM'] * 100,
}

print("=" * 80)
print("ADMM Input Group vs Feature-Selection-Benchmark Methods")
print("All results averaged over m ∈ {2,4,8,16,32,64,128,256,512,1024,2048}")
print("=" * 80)

# Create comparison table
df = pd.DataFrame(benchmark_methods).T
df = df.sort_values('RING+XOR+SUM', ascending=False)

print("\nBest-k Score Comparison (higher is better):")
print("-" * 80)
print(df.to_string(float_format=lambda x: f'{x:.1f}'))

# Add average column
df['AVG'] = df.mean(axis=1)
df = df.sort_values('AVG', ascending=False)

print("\n\nRanked by Overall Average:")
print("-" * 80)
print(df[['AVG', 'RING', 'XOR', 'RING+XOR', 'RING+XOR+SUM']].to_string(float_format=lambda x: f'{x:.1f}'))

# Highlight our method position
print("\n\n" + "=" * 80)
print("ADMM Input Group Ranking:")
print("=" * 80)
for col in ['RING', 'XOR', 'RING+XOR', 'RING+XOR+SUM', 'AVG']:
    our_score = df.loc['ADMM Input Group (Ours)', col]
    rank = (df[col] >= our_score).sum()
    total = len(df)
    print(f"{col:>15}: {our_score:>6.1f}% - Rank #{rank}/{total}")
