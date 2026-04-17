"""Parse existing benchmark result files and display comparison table at m=128."""
import os, re
from collections import defaultdict

RESULTS_DIR = r"c:\Users\12425\Documents\Projects\NEW_Pruning_20251110\Feature-Selection-Benchmark\results"

# Datasets we care about
DATASETS = ["xor", "ring", "ring+xor", "ring+xor+sum"]
TARGET_M = 128
N_SAMPLES = 1000

# Collect: results[method][dataset] = best_k_value
results = defaultdict(dict)

for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith(f"-{N_SAMPLES}.txt"):
        continue

    # Parse method and dataset from filename: method-dataset-1000.txt
    base = fname[:-len(f"-{N_SAMPLES}.txt")]
    # Find which dataset suffix matches
    matched_ds = None
    for ds in sorted(DATASETS, key=len, reverse=True):  # longest first
        if base.endswith(f"-{ds}"):
            matched_ds = ds
            method = base[:-len(f"-{ds}")]
            break
    if matched_ds is None:
        continue

    fpath = os.path.join(RESULTS_DIR, fname)
    with open(fpath, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        continue

    # Parse header to find bestK column
    header = lines[0].strip().split("\t")
    bestk_col = None
    for i, col in enumerate(header):
        if col.endswith("_bestK") and not col.endswith("_bestK2"):
            bestk_col = i
            break

    if bestk_col is None:
        continue

    # Find row for m=128
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if not parts:
            continue
        row_name = parts[0]  # e.g. "ring_128_1000"
        # Check if this row is for m=128
        if f"_{TARGET_M}_{N_SAMPLES}" in row_name:
            try:
                val = float(parts[bestk_col])
                results[method][matched_ds] = val
            except (IndexError, ValueError):
                pass

# Sort methods by average across available datasets
method_avgs = {}
for method, ds_vals in results.items():
    vals = [v for ds, v in ds_vals.items() if ds in DATASETS]
    method_avgs[method] = sum(vals) / len(vals) if vals else -1

sorted_methods = sorted(method_avgs.keys(), key=lambda m: -method_avgs[m])

# Print table
print(f"\n{'='*80}")
print(f"  best-k (%) at m={TARGET_M}, n_samples={N_SAMPLES}    (from existing result files)")
print(f"{'='*80}")
header = f"{'Method':<28s}"
for ds in DATASETS:
    header += f"  {ds:>12s}"
header += "    AVG"
print(header)
print("-" * (28 + 14 * len(DATASETS) + 7))

for method in sorted_methods:
    row = f"{method:<28s}"
    vals = []
    for ds in DATASETS:
        v = results[method].get(ds)
        if v is not None:
            row += f"  {v:>11.1%}"
            vals.append(v)
        else:
            row += f"  {'—':>12s}"
    avg = sum(vals) / len(vals) if vals else 0
    row += f"   {avg:>5.1%}"
    print(row)

print(f"\nTotal methods with m=128 data: {len(sorted_methods)}")
