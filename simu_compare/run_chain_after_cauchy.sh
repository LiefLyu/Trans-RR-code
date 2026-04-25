#!/bin/bash
# Sequential chain that takes over once cauchy finishes:
# cauchy(done) -> mix -> AR(1) -> sensitivity_cv_loss -> wait for heatmap + realdata MAE -> generate figures -> copy to plot/
set -u

EXP=/Users/lvlingfeng/Desktop/Trans-ridge/experiment
RES=$EXP/simu_compare/res
PY=/Users/lvlingfeng/Desktop/Trans-ridge/.venv/bin/python
PLOT=/Users/lvlingfeng/Desktop/Trans-ridge/ustc-overleaf-git/plot

set_env() {
    export VECLIB_MAXIMUM_THREADS=1
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
}

set_env

echo "[chain] $(date '+%H:%M:%S') waiting for cauchy JSON..."
until [ -f "$RES/2cauchy_cv_results_p400_simu1000.json" ]; do sleep 30; done
echo "[chain] $(date '+%H:%M:%S') cauchy done"

# --- mix ---
echo "[chain] $(date '+%H:%M:%S') launching mix"
cd "$EXP/simu_compare" && "$PY" -u run_mix_main.py 2>&1 | tee /tmp/mix_main.log
echo "[chain] $(date '+%H:%M:%S') mix done"

# --- AR(1) ---
echo "[chain] $(date '+%H:%M:%S') launching AR(1)"
cd "$EXP/simu_compare" && "$PY" -u run_corr_ar1_main.py 2>&1 | tee /tmp/ar1.log
echo "[chain] $(date '+%H:%M:%S') AR(1) done"

# --- sensitivity_cv_loss ---
echo "[chain] $(date '+%H:%M:%S') launching sensitivity_cv_loss"
cd "$EXP/simu_compare" && "$PY" -u run_sensitivity_cv_loss.py 2>&1 | tee /tmp/sens_cv.log
echo "[chain] $(date '+%H:%M:%S') sensitivity_cv_loss done"

# --- Wait for concurrently-running heatmap (already launched separately) ---
echo "[chain] $(date '+%H:%M:%S') waiting for heatmap JSON..."
until [ -f "$RES/2sensitivity_delta_eta_p400_simu200.json" ]; do sleep 30; done
echo "[chain] $(date '+%H:%M:%S') heatmap also done"

# --- Wait for realdata MAE (running concurrently in its own chain) ---
echo "[chain] $(date '+%H:%M:%S') waiting for realdata MAE JSON (last config)..."
until [ -f "$EXP/realdata/res/realdata_rmse_no_whiten_mae.json" ]; do sleep 30; done
echo "[chain] $(date '+%H:%M:%S') realdata MAE done"

# --- Figure generation ---
echo "[chain] $(date '+%H:%M:%S') generating figures"
cd "$EXP/simu_compare" && "$PY" -u make_methods_comparison.py 2>&1 || echo "[chain] WARN: make_methods_comparison failed"
cd "$EXP/simu_compare" && "$PY" -u make_methods_comparison_corr.py 2>&1 || echo "[chain] WARN: make_methods_comparison_corr failed"
cd "$EXP/realdata" && "$PY" -u make_boxplot.py 2>&1 || echo "[chain] WARN: make_boxplot failed"

# --- Copy figures to ustc-overleaf-git/plot/ ---
mkdir -p "$PLOT"
cp -v "$EXP/simu_compare/res/methods_comparison.pdf"      "$PLOT/methods_comparison.pdf"      2>&1 || echo "[chain] FAIL copy fig 1"
cp -v "$EXP/simu_compare/res/methods_comparison_corr.pdf" "$PLOT/methods_comparison_corr.pdf" 2>&1 || echo "[chain] FAIL copy fig 2"
cp -v "$EXP/realdata/res/realdata_boxplot.pdf"            "$PLOT/realdata_boxplot.pdf"        2>&1 || echo "[chain] FAIL copy fig 3"

echo "[chain] $(date '+%H:%M:%S') ALL_SIMS_AND_FIGURES_DONE_READY_FOR_LATEX_FILL"
