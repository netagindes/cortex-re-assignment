#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${API_URL:=http://localhost:8000}"
API_URL="$API_URL" streamlit run app/ui/streamlit_app.py
