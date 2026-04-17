@echo off
REM 快速查看 benchmark 进度
ssh -p 28128 root@hz-4.matpool.com "echo === Benchmark 进度 === && cat /root/benchmark_run/hourly_log.txt && echo. && echo 实时进程：&& ps aux | grep run_full | grep -v grep | awk '{print \"CPU: \"\$3\"%, MEM: \"\$4\"%, TIME: \"\$10}'"
pause
