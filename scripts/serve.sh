#!/usr/bin/env bash
# Start the OCR service on the configured host/port (default 127.0.0.1:8100).
# Frees the port first, then runs the single-worker app (models load once).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${OCR_PORT:-8100}"
fuser -k "${PORT}/tcp" 2>/dev/null || true

exec .venv/bin/python -m app.main
