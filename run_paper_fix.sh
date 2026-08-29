#!/usr/bin/env bash
# run_paper_fix.sh
# Runs all experiments to address reviewer concerns, inside tmux.
# Estimated runtime: ~14 hours on a single GPU.
#
# Phase 1: Generate preliminary figures (uses existing data, ~2 min)
# Phase 2: Full evaluation suite with clean splits (paper_eval_suite.py, ~14h)
# Phase 3: Regenerate final figures including clean results (~2 min)

set -euo pipefail
cd /home/ubuntu/rlvr-vectors

VENV=/home/ubuntu/rlvr-vectors/.venv
LOG=outputs/paper_fix_run.log

echo "=== PAPER FIX RUN STARTED: $(date) ===" | tee -a "$LOG"

# ── Activate virtualenv ────────────────────────────────────────────────────────
source "$VENV/bin/activate"

# ── Phase 1: Preliminary figures from existing data ────────────────────────────
echo "" | tee -a "$LOG"
echo "[PHASE 1] Generating preliminary figures..." | tee -a "$LOG"
python3 generate_figures.py 2>&1 | tee -a "$LOG"
echo "[PHASE 1] Done." | tee -a "$LOG"

# ── Phase 2: Full clean-split evaluation suite ─────────────────────────────────
echo "" | tee -a "$LOG"
echo "[PHASE 2] Starting paper_eval_suite.py  (estimated ~14 hours)" | tee -a "$LOG"
echo "          Splits: CALIB=450-499, VAL=400-449, TEST=0-399" | tee -a "$LOG"
python3 paper_eval_suite.py 2>&1 | tee -a "$LOG"
echo "[PHASE 2] Done." | tee -a "$LOG"

# ── Phase 3: Regenerate figures with clean results ─────────────────────────────
echo "" | tee -a "$LOG"
echo "[PHASE 3] Regenerating final figures with clean results..." | tee -a "$LOG"
python3 generate_figures.py 2>&1 | tee -a "$LOG"
echo "[PHASE 3] Done." | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
echo "Results in: outputs/paper_eval_stats.json"
echo "Figures in: outputs/fig{1,2,3,4}_*.png"
