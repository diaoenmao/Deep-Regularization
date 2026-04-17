@echo off
REM Full Experiment Runner for Pruning Methods
REM Estimated time: ~10 hours on GPU, ~20+ hours on CPU

echo ============================================================
echo       FULL PRUNING EXPERIMENT - CNN3 on MNIST
echo ============================================================
echo.
echo This will run comprehensive experiments across:
echo   - 9 optimizers: ADMM/Ppercent/Lasso x global/layer/neuron
echo   - 3 score types: first order, second order, combined
echo   - 10 parameter values per optimizer
echo   - Total: 270 experiments
echo.
echo Estimated time: 8-12 hours (GPU) / 20+ hours (CPU)
echo.

REM Check for GPU
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
echo.

set /p DEVICE="Use GPU (cuda) or CPU? [cuda/cpu]: "
set /p EPOCHS="Number of epochs (default=10, quick=3): "

if "%EPOCHS%"=="" set EPOCHS=10

echo.
echo Starting experiment with device=%DEVICE%, epochs=%EPOCHS%
echo Results will be saved to: results/metrics/
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause > nul

REM Create output directories
if not exist "results\metrics" mkdir results\metrics
if not exist "results\plots" mkdir results\plots

REM Run experiment
python run_full_experiment.py --device %DEVICE% --epochs %EPOCHS%

echo.
echo ============================================================
echo Experiment completed! Generating figures...
echo ============================================================

REM Generate plots
python plot_big_figure.py

echo.
echo ============================================================
echo ALL DONE!
echo ============================================================
echo Results saved to:
echo   - results/metrics/cnn3_MNIST_all_optimizers_experiment_results.json
echo   - results/plots/big_figure_3x3.png
echo   - results/plots/big_figure_3x3.pdf
echo   - results/plots/score_comparison.png
echo   - results/plots/pareto_frontier.png
echo   - results/plots/summary_bars.png
echo ============================================================

pause
