#!/bin/bash
# 监控 benchmark 运行进度
# 用法：bash monitor.sh

SERVER="hz-4.matpool.com"
PORT="28128"
USER="root"

echo "=========================================="
echo "Benchmark 运行进度监控"
echo "时间：$(date)"
echo "=========================================="

# 检查进程
echo ""
echo "[进程状态]"
ssh -p $PORT $USER@$SERVER "ps aux | grep run_full | grep -v grep | awk '{print \"CPU: \"\$3\"%, MEM: \"\$4\"%, TIME: \"\$10}'"

# 检查结果文件
echo ""
echo "[结果文件]"
ssh -p $PORT $USER@$SERVER "ls -lh /root/benchmark_run/benchmark_with_tracking.json | awk '{print \$5, \$9}'"

# 显示完成的进度
echo ""
echo "[完成进度]"
ssh -p $PORT $USER@$SERVER "/root/miniconda3/envs/myconda/bin/python -c \"
import json
d = json.load(open('/root/benchmark_run/benchmark_with_tracking.json'))
exps = d['experiments']
total_dims = sum(len(v['dimensions']) for v in exps.values())
print(f'已完成数据集：{list(exps.keys())}')
print(f'已完成维度数：{total_dims}/40')
for ds, data in exps.items():
    dims = list(data['dimensions'].keys())
    avg_bk = [data['dimensions'][m].get('best_k', 0) for m in dims]
    if avg_bk:
        print(f'  {ds}: {len(dims)}个维度, 平均 best-k: {sum(avg_bk)/len(avg_bk):.2%}')
\""

echo ""
echo "=========================================="
