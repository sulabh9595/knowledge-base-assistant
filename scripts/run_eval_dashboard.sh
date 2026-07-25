#!/usr/bin/env bash
# Launcher for the DeepEval evaluation dashboard.
#
# Runs the Streamlit dashboard on port 8502 so it does not collide with
# the main frontend (frontend/app.py), which stays on the default 8501.
#
# Usage:
#   ./scripts/run_eval_dashboard.sh
#   PORT=9000 ./scripts/run_eval_dashboard.sh   # override port
#
# Stop with Ctrl-C.

set -euo pipefail

# Resolve project root from this script's location (scripts/ is one level below root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default port; allow override via environment variable.
PORT="${PORT:-8502}"

cd "${PROJECT_ROOT}"

# Prepend the project root to PYTHONPATH so Python resolves `app.config...`
# to the real top-level `app/` package, not to the sibling frontend/app.py
# (which lives next to this dashboard and would otherwise shadow the import).
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting DeepEval evaluation dashboard on http://localhost:${PORT}"
exec streamlit run frontend/eval_dashboard.py \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false
