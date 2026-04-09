@echo off
REM Quick GPU Test - Runs all ablations in quick mode (~5 min total)
REM Run from: E:\Projects\NEW_Pruning_20251110\custom_admm

cd /d E:\Projects\NEW_Pruning_20251110\custom_admm

echo ============================================================
echo QUICK GPU TEST - ~5 MINUTES TOTAL
echo ============================================================

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo [1/3] Polynomial (quick)...
python run_polynomial_ablation.py --device cuda --quick

echo.
echo [2/3] Iterative (quick)...
python run_iterative_ablation.py --device cuda --quick

echo.
echo [3/3] Transformer Pretrain (quick)...
python run_transformer_pretrain_ablation.py --device cuda --quick

echo.
echo ============================================================
echo DONE - Check results in custom_admm/results/
echo ============================================================

pause