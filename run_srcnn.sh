#!/usr/bin/env bash
# run_srcnn.sh — run SRCNN training/evaluation once
# Assumptions:
#   - Parameters (data paths, epochs, loss, etc.) are defined inside SRCNN/srcnn.py

set -euo pipefail

PYTHON="${PYTHON:-python}"

log(){ printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*"; }

# Check Python and install requirements if present
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python not found"; exit 1; }
if [ -f requirements.txt ]; then
  log "Installing requirements (if needed)"
  "$PYTHON" -m pip install -r requirements.txt >/dev/null
fi

cd code/SRCNN

# Create CSV files for srcnn.py
log "Creating CSV files for SRCNN"
"$PYTHON" create_csv.py

# Run SRCNN (single pass; all params inside srcnn.py)
if [ -f "srcnn.py" ]; then
  log "Running srcnn.py"
  "$PYTHON" srcnn.py
  log "Done. Check result_data_*, result_plot_* and outputs/ for artifacts."
else
  echo "srcnn.py not found."
  exit 1
fi


NOTEBOOK_IN="analysis.ipynb"

echo "Executing analysis notebook..."
jupyter nbconvert --to notebook --execute "$NOTEBOOK_IN"

echo "Done. Plots in result_plot_fits/"
