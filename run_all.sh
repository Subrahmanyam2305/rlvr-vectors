#!/usr/bin/env bash
# run_all.sh — full experiment pipeline after all reviewer fixes.
#
# Estimated runtimes (single GPU, Qwen2.5-1.5B):
#   Gate analysis + spectral null : ~1 hour
#   paper_eval_suite (clean eval) : ~14 hours
#   comprehensive_suite           : ~20 hours
#   Total                         : ~35 hours
#
# Monitor: tmux attach -t paper_fix

set -euo pipefail
cd /home/ubuntu/rlvr-vectors
source .venv/bin/activate
LOG=outputs/run_all.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== RUN ALL STARTED ==="

# Phase 1: Figures from existing data (fast, ~2 min)
log "[PHASE 1] Preliminary figures from existing data..."
python3 generate_figures.py 2>&1 | tee -a "$LOG"
log "[PHASE 1] Done."

# Phase 2: Gate analysis + spectral null (~1 hour)
log "[PHASE 2] Gate analysis + spectral null distributions..."
python3 gate_analysis.py 2>&1 | tee -a "$LOG"
log "[PHASE 2] Done."

# Phase 3: Core paper eval suite with all reviewer fixes (~14 hours)
log "[PHASE 3] Paper eval suite (clean splits, fixed controls, n=400 test)..."
python3 paper_eval_suite.py 2>&1 | tee -a "$LOG"
log "[PHASE 3] Done."

# Phase 4: Regenerate figures with clean results
log "[PHASE 4] Regenerating figures with clean results..."
python3 generate_figures.py 2>&1 | tee -a "$LOG"
log "[PHASE 4] Done."

# Phase 5: Comprehensive suite (multi-source, multi-target, ablations, GSM8K) (~20 hours)
log "[PHASE 5] Comprehensive suite..."
python3 comprehensive_suite.py 2>&1 | tee -a "$LOG"
log "[PHASE 5] Done."

log "=== ALL DONE ==="
log "Key outputs:"
log "  outputs/gate_analysis.json          — per-projection gating stats"
log "  outputs/spectral_null.json          — shape-matched empirical nulls"
log "  outputs/paper_eval_stats.json       — main results + null distributions"
log "  outputs/comprehensive_results.json  — multi-source/target + ablations"
log "  outputs/fig{1,2,3,4}_*.png          — paper figures"
