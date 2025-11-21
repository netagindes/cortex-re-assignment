#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

: "${API_URL:=http://localhost:8000}"

if [[ -x "$PROJECT_ROOT/.venv/bin/streamlit" ]]; then
    STREAMLIT_BIN="$PROJECT_ROOT/.venv/bin/streamlit"
else
    STREAMLIT_BIN="$(command -v streamlit || true)"
fi

if [[ -z "${STREAMLIT_BIN}" ]]; then
    echo "Streamlit executable not found. Install dependencies or create .venv first." >&2
    exit 1
fi

API_URL="$API_URL" "$STREAMLIT_BIN" run app/ui/streamlit_app.py
