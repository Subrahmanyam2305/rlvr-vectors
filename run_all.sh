#!/usr/bin/env bash
# run_all.sh — full experiment pipeline, all reviewer fixes applied.
#
# Estimated runtimes on a single GPU (Qwen2.5-1.5B):
#   Phase 1  Figures from existing data         ~2 min
#   Phase 2  Gate analysis + spectral null       ~1 hour
#   Phase 3  Gate mediation (causal test)        ~4 hours
#   Phase 4  paper_eval_suite (clean, n=400)     ~14 hours
#   Phase 5  Regenerate final figures            ~2 min
#   Phase 6  Comprehensive suite                 ~20 hours
#   Total                                        ~40 hours
#
# Monitor: tmux attach -t paper_fix

set -euo pipefail
cd "$(dirname "$0")"    # always run from script's own directory
source .venv/bin/activate
LOG=outputs/run_all.log
mkdir -p outputs

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== RUN ALL STARTED ==="

log "[PHASE 1] Preliminary figures from existing data..."
python3 generate_figures.py 2>&1 | tee -a "$LOG"
log "[PHASE 1] Done."

log "[PHASE 2] Gate analysis (paired prompt-level) + spectral null (power iter)..."
python3 gate_analysis.py 2>&1 | tee -a "$LOG"
log "[PHASE 2] Done."

log "[PHASE 3] Gate mediation — causal test of gate-mismatch hypothesis..."
python3 gate_mediation.py 2>&1 | tee -a "$LOG"
log "[PHASE 3] Done."

log "[PHASE 4] Paper eval suite (clean splits, fixed controls, n=400 test)..."
python3 paper_eval_suite.py 2>&1 | tee -a "$LOG"
log "[PHASE 4] Done."

log "[PHASE 5] Regenerating final figures with clean results..."
python3 generate_figures.py 2>&1 | tee -a "$LOG"
log "[PHASE 5] Done."

log "[PHASE 6] Comprehensive suite (multi-source, multi-target, ablations, GSM8K)..."
python3 comprehensive_suite.py 2>&1 | tee -a "$LOG"
log "[PHASE 6] Done."

log "=== ALL DONE ==="
log "Key outputs:"
log "  outputs/gate_analysis.json          — paired prompt-level gating stats"
log "  outputs/spectral_null.json          — shape-matched empirical nulls"
log "  outputs/gate_mediation_results.json — causal gate mediation"
log "  outputs/paper_eval_stats.json       — main results + null distributions"
log "  outputs/comprehensive_results.json  — multi-source/target + ablations"
log "  outputs/fig{1,2,3,4}_*.png          — paper figures"
