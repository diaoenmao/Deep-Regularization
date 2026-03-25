@echo off
REM 下载 benchmark 结果文件
echo 正在下载 benchmark_with_tracking.json...
scp -P 28128 root@hz-4.matpool.com:/root/benchmark_run/benchmark_with_tracking.json ./benchmark_results_20260315.json
if %ERRORLEVEL% EQU 0 (
    echo 下载成功!
    dir benchmark_results_20260315.json
) else (
    echo 下载失败，请手动输入密码
)
pause
