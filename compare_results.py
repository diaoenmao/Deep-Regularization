#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare old vs new benchmark results."""
import pandas as pd
import numpy as np

# Old results (Mar 6) - from results_backup
old_xor = pd.DataFrame({
    'Dataset': ['xor_2_1000', 'xor_4_1000', 'xor_8_1000', 'xor_16_1000', 'xor_32_1000',
                'xor_64_1000', 'xor_128_1000', 'xor_256_1000', 'xor_512_1000', 'xor_1024_1000', 'xor_2048_1000'],
    'bestK': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666666, 0.3333333333333333, 0.16666666666666666],
})

old_ring = pd.DataFrame({
    'Dataset': ['ring_8_1000', 'ring_16_1000', 'ring_32_1000', 'ring_64_1000', 'ring_128_1000',
                'ring_256_1000', 'ring_512_1000', 'ring_1024_1000', 'ring_2048_1000'],
    'bestK': [1.0, 1.0, 1.0, 0.9166666666666666, 0.5, 0.5, 0.5, 0.16666666666666666, 0.16666666666666666],
})

old_rx = pd.DataFrame({
    'Dataset': ['ring+xor_4_1000', 'ring+xor_8_1000', 'ring+xor_16_1000', 'ring+xor_32_1000',
                'ring+xor_64_1000', 'ring+xor_128_1000', 'ring+xor_256_1000', 'ring+xor_512_1000',
                'ring+xor_1024_1000', 'ring+xor_2048_1000'],
    'bestK': [1.0, 0.75, 0.6666666666666666, 0.625, 0.5416666666666667, 0.6666666666666666,
              0.625, 0.2916666666666667, 0.16666666666666666, 0.08333333333333333],
})

old_rxsum = pd.DataFrame({
    'Dataset': ['ring+xor+sum_6_1000', 'ring+xor+sum_8_1000', 'ring+xor+sum_16_1000',
                'ring+xor+sum_32_1000', 'ring+xor+sum_64_1000', 'ring+xor+sum_128_1000',
                'ring+xor+sum_256_1000', 'ring+xor+sum_512_1000', 'ring+xor+sum_1024_1000',
                'ring+xor+sum_2048_1000'],
    'bestK': [1.0, 0.8888888888888888, 0.7222222222222222, 0.6944444444444443, 0.6666666666666666,
              0.6388888888888888, 0.6944444444444443, 0.5555555555555555, 0.3888888888888889, 0.3333333333333333],
})

# New results (Mar 19) - parallel run on A4000
new_xor = pd.DataFrame({
    'Dataset': ['xor_2_1000', 'xor_4_1000', 'xor_8_1000', 'xor_16_1000', 'xor_32_1000',
                'xor_64_1000', 'xor_128_1000', 'xor_256_1000', 'xor_512_1000', 'xor_1024_1000', 'xor_2048_1000'],
    'bestK': [1.0, 0.75, 0.16666666666666666, 0.08333333333333333, 0.0, 0.0,
              0.3333333333333333, 0.3333333333333333, 0.8333333333333334, 0.4166666666666667, 0.4166666666666667],
})

new_ring = pd.DataFrame({
    'Dataset': ['ring_8_1000', 'ring_16_1000', 'ring_32_1000', 'ring_64_1000', 'ring_128_1000',
                'ring_256_1000', 'ring_512_1000', 'ring_1024_1000', 'ring_2048_1000'],
    'bestK': [0.3333333333333333, 0.0, 0.0, 0.4166666666666667, 0.0,
              0.16666666666666666, 0.0, 0.16666666666666666, 0.08333333333333333],
})

new_rx = pd.DataFrame({
    'Dataset': ['ring+xor_4_1000', 'ring+xor_8_1000', 'ring+xor_16_1000', 'ring+xor_32_1000',
                'ring+xor_64_1000', 'ring+xor_128_1000', 'ring+xor_256_1000', 'ring+xor_512_1000',
                'ring+xor_1024_1000', 'ring+xor_2048_1000'],
    'bestK': [1.0, 0.5, 0.20833333333333334, 0.125, 0.3333333333333333, 0.375,
              0.3333333333333333, 0.16666666666666666, 0.0, 0.0],
})

new_rxsum = pd.DataFrame({
    'Dataset': ['ring+xor+sum_6_1000', 'ring+xor+sum_8_1000', 'ring+xor+sum_16_1000',
                'ring+xor+sum_32_1000', 'ring+xor+sum_64_1000', 'ring+xor+sum_128_1000',
                'ring+xor+sum_256_1000', 'ring+xor+sum_512_1000', 'ring+xor+sum_1024_1000',
                'ring+xor+sum_2048_1000'],
    'bestK': [1.0, 0.8055555555555557, 0.4444444444444444, 0.47222222222222215, 0.5277777777777777,
              0.611111111111111, 0.5277777777777778, 0.4444444444444444, 0.4444444444444444, 0.3333333333333333],
})

print("=" * 80)
print("COMPARISON: New Run (Mar 19, A4000 Parallel) vs Old Backup (Mar 6)")
print("=" * 80)

def compare_datasets(name, old_df, new_df):
    old_df = old_df.copy()
    new_df = new_df.copy()

    old_df['m'] = old_df['Dataset'].apply(lambda x: int(x.split('_')[1]))
    new_df['m'] = new_df['Dataset'].apply(lambda x: int(x.split('_')[1]))

    merged = pd.merge(old_df, new_df, on='m', how='outer', suffixes=('_old', '_new'))

    print(f"\n{name}:")
    print(f"{'m':>8} | {'Old bestK':>10} | {'New bestK':>10} | {'Diff':>8}")
    print("-" * 55)

    for _, row in merged.sort_values('m').iterrows():
        old_k = row['bestK_old'] if pd.notna(row['bestK_old']) else 0
        new_k = row['bestK_new'] if pd.notna(row['bestK_new']) else 0
        diff = new_k - old_k
        diff_str = f"{diff:+.1%}" if diff != 0 else "  0.0%"
        marker = "↓" if diff < 0 else ("↑" if diff > 0 else " ")
        print(f"{int(row['m']):>8} | {old_k:>10.1%} | {new_k:>10.1%} | {marker} {diff_str:>6}")

    old_avg = merged['bestK_old'].mean()
    new_avg = merged['bestK_new'].mean()
    print(f"\nAverage: Old={old_avg:.1%}, New={new_avg:.1%}, Diff={new_avg-old_avg:+.1%}")
    return old_avg, new_avg

xor_old, xor_new = compare_datasets("XOR (k=2)", old_xor, new_xor)
ring_old, ring_new = compare_datasets("RING (k=2)", old_ring, new_ring)
rx_old, rx_new = compare_datasets("RING+XOR (k=4)", old_rx, new_rx)
rxsum_old, rxsum_new = compare_datasets("RING+XOR+SUM (k=6)", old_rxsum, new_rxsum)

print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print(f"{'Dataset':<20} | {'Old Avg':>10} | {'New Avg':>10} | {'Diff':>10}")
print("-" * 60)
print(f"{'XOR':<20} | {xor_old:>10.1%} | {xor_new:>10.1%} | {xor_new-xor_old:>+10.1%}")
print(f"{'RING':<20} | {ring_old:>10.1%} | {ring_new:>10.1%} | {ring_new-ring_old:>+10.1%}")
print(f"{'RING+XOR':<20} | {rx_old:>10.1%} | {rx_new:>10.1%} | {rx_new-rx_old:>+10.1%}")
print(f"{'RING+XOR+SUM':<20} | {rxsum_old:>10.1%} | {rxsum_new:>10.1%} | {rxsum_new-rxsum_old:>+10.1%}")
print("-" * 60)
overall_old = (xor_old + ring_old + rx_old + rxsum_old) / 4
overall_new = (xor_new + ring_new + rx_new + rxsum_new) / 4
print(f"{'OVERALL':<20} | {overall_old:>10.1%} | {overall_new:>10.1%} | {overall_new-overall_old:>+10.1%}")
