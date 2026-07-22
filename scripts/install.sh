#!/usr/bin/env bash
# One-command environment bring-up for the LOCAL CPU deployment (ARM64 / x86).
#
#   bash scripts/install.sh
#
# Installs system build tools, creates .venv, installs PaddlePaddle (CPU) and the
# Python deps (incl. llama-cpp-python compiled with CPU SIMD). No GPU, no CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> system packages (build tools + OpenCV/Paddle runtime libs)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential cmake pkg-config python3-venv python3-dev \
  libgl1 libglib2.0-0 libgomp1 git-lfs

echo "==> python venv"
python3 -m venv .venv
.venv/bin/python -m pip install -U pip wheel setuptools -q

echo "==> python dependencies (this compiles llama-cpp-python — a few minutes)"
# Build llama.cpp with native CPU SIMD (ARM NEON / AVX) for best throughput.
# The default reader (RapidOCR/ONNX) and the Qwen extractor come from here.
CMAKE_ARGS="${CMAKE_ARGS:--DGGML_NATIVE=ON}" \
  .venv/bin/python -m pip install --timeout 1000 --retries 10 -r requirements.txt

# OPTIONAL: PaddlePaddle CPU + PaddleOCR, only for the paddle reader engines
# (OCR_READER_ENGINE=paddleocr-vl|paddleocr). Skipped by default because
# paddle's native CPU inference segfaults on this aarch64 build; set
# INSTALL_PADDLE=1 to install it on a host where paddle works.
if [ "${INSTALL_PADDLE:-0}" = "1" ]; then
  PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cpu/}"
  echo "==> (optional) PaddlePaddle (CPU) from $PADDLE_INDEX"
  .venv/bin/python -m pip install --timeout 1000 --retries 10 paddlepaddle -i "$PADDLE_INDEX" \
    || .venv/bin/python -m pip install --timeout 1000 --retries 10 paddlepaddle
  .venv/bin/python -m pip install --timeout 1000 --retries 10 "paddleocr>=3.2.0" "paddlex[ocr]>=3.2.0"
fi

echo
echo "==> install complete. Next:"
echo "    bash scripts/download_models.sh    # fetch Qwen3-8B GGUF + PaddleOCR-VL weights"
echo "    bash scripts/serve.sh              # start the service on 127.0.0.1:8100"
