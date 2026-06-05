#!/usr/bin/env bash
# Launch the translation review UI.
# Usage:  ./run_review.sh [port]   (default: 8520)
#
# If streamlit + pandas are importable in the current python, uses that.
# Otherwise creates a local venv at ./.venv and installs requirements.txt.
# Works around PEP 668 / externally-managed Python on Debian/Ubuntu.
#
# Env vars:
#   REVIEW_DATA_DIR  — override CSV folder (default: same folder as this script)
#   REVIEW_PYTHON    — python to use (default: python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8520}"
PY="${REVIEW_PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: $PY not found. Install Python 3.10+ and retry." >&2
    exit 1
fi

if "$PY" -c "import streamlit, pandas" 2>/dev/null; then
    RUN_PY="$PY"
    echo "==> using existing $PY (streamlit + pandas already installed)"
else
    VENV="$SCRIPT_DIR/.venv"
    if [[ ! -d "$VENV" ]]; then
        echo "==> creating venv at $VENV"
        "$PY" -m venv "$VENV"
    fi
    RUN_PY="$VENV/bin/python"
    echo "==> installing deps into venv"
    "$RUN_PY" -m pip install --quiet --disable-pip-version-check --upgrade pip
    "$RUN_PY" -m pip install --quiet --disable-pip-version-check \
        -r "$SCRIPT_DIR/requirements.txt"
fi

echo "==> open http://localhost:$PORT in your browser"
exec "$RUN_PY" -m streamlit run "$SCRIPT_DIR/review_app.py" \
    --server.port="$PORT" \
    --server.headless=true \
    --browser.gatherUsageStats=false
