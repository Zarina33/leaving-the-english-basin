#!/usr/bin/env bash
# Run all remaining experiments sequentially.
# Each step's failure is logged but does NOT stop subsequent steps.
# Monitor:   tail -f logs/remaining.log

set +e
mkdir -p logs

PY="python -u"
LOG="logs/remaining.log"

run() {
  local name="$1"; shift
  echo                                          | tee -a "$LOG"
  echo "===================== $name ====================="  | tee -a "$LOG"
  date -Iseconds                                | tee -a "$LOG"
  eval "$@" 2>&1                                | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "===== EXIT $rc -- $name ====="          | tee -a "$LOG"
  return $rc
}

run "33-extract-new-models"   "$PY scripts/33_extract_models.py --model all"
run "35-analyze-llama32_3b"   "$PY scripts/35_analyze_model.py --model llama32_3b"
run "35-analyze-qwen25_7b"    "$PY scripts/35_analyze_model.py --model qwen25_7b"
run "35-analyze-olmo2_7b"     "$PY scripts/35_analyze_model.py --model olmo2_7b"
run "34-extract-mbert"        "$PY scripts/34_extract_mbert.py"
run "35-analyze-mbert"        "$PY scripts/35_analyze_model.py --model mbert"
run "36-check-llama-cka"      "$PY scripts/36_check_llama_cka.py"
run "37-permutation-10k"      "$PY scripts/37_permutation_10k.py"

echo                                            | tee -a "$LOG"
echo "==== ALL EXPERIMENTS FINISHED ===="       | tee -a "$LOG"
date -Iseconds                                  | tee -a "$LOG"
