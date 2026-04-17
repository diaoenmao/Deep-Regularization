@echo off
REM GPU Ablation Experiments Runner
REM RTX 3060 Ti (8GB VRAM)
REM Run from: E:\Projects\NEW_Pruning_20251110\custom_admm

cd /d E:\Projects\NEW_Pruning_20251110\custom_admm

echo ============================================================
echo GPU ABLATION EXPERIMENTS - RTX 3060 Ti
echo ============================================================
echo.

REM Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
if errorlevel 1 (
    echo ERROR: PyTorch/CUDA not available
    pause
    exit /b 1
)

echo.
echo ============================================================
echo PHASE 1: POLYNOMIAL ABLATION (~15 min)
echo ============================================================
echo Comparing: group vs expanded vs hierarchical selection
echo.
python run_polynomial_ablation.py --device cuda --full
if errorlevel 1 (
    echo WARNING: Polynomial ablation failed
)

echo.
echo ============================================================
echo PHASE 2: ITERATIVE ABLATION (~20 min)
echo ============================================================
echo Comparing: single_pass vs iterative_hard vs gradual_admm
echo.
python run_iterative_ablation.py --device cuda --full
if errorlevel 1 (
    echo WARNING: Iterative ablation failed
)

echo.
echo ============================================================
echo PHASE 3: TRANSFORMER PRETRAIN ABLATION (~30 min)
echo ============================================================
echo Comparing: mlp_baseline vs transformer_pretrain
echo.
python run_transformer_pretrain_ablation.py --device cuda --full
if errorlevel 1 (
    echo WARNING: Transformer pretrain ablation failed
)

echo.
echo ============================================================
echo ALL EXPERIMENTS COMPLETE
echo ============================================================
echo.
echo Results saved to: %cd%\results\
echo.
dir results\*.json /o-d | findstr "ablation"
echo.

pause