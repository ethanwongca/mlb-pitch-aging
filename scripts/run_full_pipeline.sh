#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNNING_UNDER_CAFFEINATE:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
  export RUNNING_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

# Run from repo root regardless of where script is launched from
cd "$(dirname "$0")/.."

CONDA_ENV_NAME="${PIPELINE_CONDA_ENV:-mlb-pitch-aging}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
  else
    echo "Conda is not available on PATH; cannot activate ${CONDA_ENV_NAME}." >&2
    exit 1
  fi
fi

mkdir -p logs
LOG_FILE="logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOG_FILE"
}

run_step() {
  local label="$1"
  local cmd="$2"
  log "START: ${label}"
  eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
  log "DONE:  ${label}"
}

log "Pipeline start"
log "Python: $(python --version 2>&1)"
log "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"

run_step "Data" "python src/data.py"
run_step "Prepare" "python src/prepare.py"
run_step "EDA Plots" "python src/eda_plots.py"

run_step "Models" "python src/models.py"
run_step "Inference" "python src/inference.py"

# Bivariate can be expensive and may already be done. Set RERUN_BIVARIATE=1 to force rerun.
if [[ "${RERUN_BIVARIATE:-0}" == "1" ]]; then
  run_step "Bivariate" "python src/bivariate.py"
else
  log "SKIP: Bivariate (set RERUN_BIVARIATE=1 to run)"
fi
 
run_step "SCG" "python src/scg.py"
run_step "Tables" "python src/tables.py"

log "Pipeline complete"
log "Log saved to ${LOG_FILE}"
