#!/usr/bin/env bash
# Launcher for the Unified Streamlit UI (including DeepEval Evaluation Dashboard).
#
# Launches the unified frontend application (frontend/app.py) containing all 6 tabs.
#
# Usage:
#   ./scripts/run_eval_dashboard.sh
#   PORT=8501 ./scripts/run_eval_dashboard.sh   # override port
#
# Stop with Ctrl-C.

set -euo pipefail

# Resolve project root from this script's location (scripts/ is one level below root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default port; allow override via environment variable.
PORT="${PORT:-8501}"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting Unified Streamlit Web UI on http://localhost:${PORT}"
exec streamlit run frontend/app.py \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false

