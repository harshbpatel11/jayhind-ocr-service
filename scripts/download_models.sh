#!/usr/bin/env bash
# Download the model weights the pipeline needs (run once, after install.sh):
#   * Qwen3-8B GGUF (Q4_K_M ≈ 4.7 GB) — the extractor LLM
#   * PaddleOCR-VL 1.6 weights           — the reader (cached under ~/.paddlex)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

QWEN_REPO="${QWEN_GGUF_REPO:-Qwen/Qwen3-8B-GGUF}"
QWEN_FILE="${QWEN_GGUF_FILE:-Qwen3-8B-Q4_K_M.gguf}"

echo "==> downloading $QWEN_REPO / $QWEN_FILE -> models/"
QWEN_REPO="$QWEN_REPO" QWEN_FILE="$QWEN_FILE" .venv/bin/python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
repo, filename = os.environ["QWEN_REPO"], os.environ["QWEN_FILE"]
src = hf_hub_download(repo_id=repo, filename=filename, local_dir="models")
target = os.path.join("models", "Qwen3-8B-Q4_K_M.gguf")
if os.path.abspath(src) != os.path.abspath(target):
    shutil.copy(src, target)
print("saved", target)
PY

# The default reader (RapidOCR/ONNX) fetches its own small PP-OCR ONNX models
# automatically on first use — nothing to download here.

# OPTIONAL: prefetch PaddleOCR-VL weights (only if the paddle engines are used).
if .venv/bin/python -c "import paddleocr" >/dev/null 2>&1; then
  echo "==> prefetching PaddleOCR-VL weights (paddle detected)"
  .venv/bin/python - <<'PY' || echo "   (VL weights will download on first use instead)"
from paddleocr import PaddleOCRVL
PaddleOCRVL()
print("PaddleOCR-VL weights ready")
PY
fi

echo "==> models ready. Start the service:  bash scripts/serve.sh"
