@echo off
REM ============================================================
REM  Setup & run the Feature-Selection-Benchmark with ADMM/Lasso
REM  methods integrated from the pruning codebase.
REM ============================================================
setlocal

set "ROOT=%~dp0"
set "BENCH=%ROOT%Feature-Selection-Benchmark"

echo ===== Installing benchmark dependencies =====
pip install numpy scipy pandas torch captum scikit-learn tqdm 2>nul

echo.
echo ===== Available ADMM / Lasso methods =====
echo   admm_global   admm_layer   admm_neuron
echo   lasso_global  lasso_layer  lasso_neuron
echo.

if "%1"=="" (
    echo Usage:  run_benchmark.bat ^<method^>
    echo Example: run_benchmark.bat admm_global
    echo.
    echo Running admm_global by default...
    set "METHOD=admm_global"
) else (
    set "METHOD=%1"
)

echo.
echo ===== Running benchmark with method: %METHOD% =====
cd /d "%BENCH%"
python main-benchmark.py %METHOD%

echo.
echo ===== Done! Results are in %BENCH%\results\ =====
endlocal
