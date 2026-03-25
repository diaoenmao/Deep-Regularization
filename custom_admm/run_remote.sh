#!/bin/bash
# Run benchmark on remote server with GPU support

cd /root/benchmark_run/custom_admm

PYTHON=/root/miniconda3/envs/myconda/bin/python

echo "=========================================="
echo "  Starting Full Benchmark with Tracking"
echo "=========================================="
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run with GPU, save output to log file
$PYTHON run_full_benchmark_with_tracking.py \
    --gpu 0 \
    --save-interval 50 \
    --output /root/benchmark_run/benchmark_with_tracking.json \
    2>&1 | tee /root/benchmark_run/benchmark.log

echo ""
echo "End time: $(date)"
echo "Results saved to: /root/benchmark_run/benchmark_with_tracking.json"
echo "Log saved to: /root/benchmark_run/benchmark.log"
