#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel benchmark runner - runs multiple dimensions concurrently.

This script spawns multiple processes to run different dimensions in parallel,
maximizing GPU utilization.

Usage:
    python run_parallel_benchmark.py --max-workers 8
"""
import sys
import os
import argparse
import subprocess
import time
from multiprocessing import Pool, cpu_count

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Dataset configuration
datasets_config = [
    ("xor",          2, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring",         2, [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor",     4, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor+sum", 6, [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
]

def get_gpu_memory():
    """Get current GPU memory usage using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        used, total = result.stdout.strip().split(", ")
        return int(used), int(total)
    except:
        return 0, 16000  # Default to 16GB if nvidia-smi fails

def run_single_dimension(args):
    """Run a single dimension benchmark."""
    ds_name, k, n_features, seed, n_samples = args
    cmd = [
        sys.executable,
        os.path.join(ROOT, "run_single_dimension.py"),
        "--dataset", ds_name,
        "--k", str(k),
        "--n-features", str(n_features),
        "--seed", str(seed),
        "--n-samples", str(n_samples),
        "--output-dir", os.path.join(ROOT, "results")
    ]
    print(f"Starting: {ds_name} m={n_features}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {ds_name} m={n_features}: {result.stderr[:200]}")
    else:
        print(f"Done: {ds_name} m={n_features}")
    return (ds_name, n_features, result.returncode == 0)

def main():
    parser = argparse.ArgumentParser(description="Parallel benchmark runner")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Maximum number of parallel workers (default: 8)")
    parser.add_argument("--gpu-memory-limit", type=int, default=14000,
                        help="GPU memory limit in MB (default: 14000 for 16GB GPU)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples (default: 1000)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Parallel ADMM Input Group Benchmark")
    print("=" * 70)
    print(f"Max workers: {args.max_workers}")
    print(f"GPU memory limit: {args.gpu_memory_limit} MB")
    print("=" * 70)

    # Create list of all tasks
    tasks = []
    for ds_name, k, dimensions in datasets_config:
        for n_features in dimensions:
            tasks.append((ds_name, k, n_features, args.seed, args.n_samples))

    print(f"Total tasks: {len(tasks)}")
    print()

    # Run tasks with memory-aware scheduling
    completed = 0
    failed = 0
    results = []

    with Pool(min(args.max_workers, len(tasks))) as pool:
        async_results = [pool.apply_async(run_single_dimension, (t,)) for t in tasks]

        for ar in async_results:
            try:
                result = ar.get(timeout=7200)  # 2 hour timeout per task
                results.append(result)
                completed += 1
                if not result[2]:  # failed
                    failed += 1

                # Print progress
                used, total = get_gpu_memory()
                print(f"Progress: {completed}/{len(tasks)} ({failed} failed), GPU: {used}/{total} MB")
            except Exception as e:
                print(f"Task failed: {e}")
                failed += 1

    # Summary
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"Completed: {completed}/{len(tasks)}")
    print(f"Failed: {failed}/{len(tasks)}")

    # Print per-dataset summary
    ds_results = {}
    for ds_name, k, dimensions in datasets_config:
        ds_tasks = [r for r in results if r[0] == ds_name]
        success = sum(1 for r in ds_tasks if r[2])
        ds_results[ds_name] = f"{success}/{len(ds_tasks)}"

    print()
    for ds, summary in ds_results.items():
        print(f"  {ds}: {summary}")

if __name__ == "__main__":
    main()
