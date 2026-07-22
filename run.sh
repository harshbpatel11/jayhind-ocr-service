#!/usr/bin/env bash
# One-command setup + run. Works on Kaggle / Colab / any Linux CUDA box.
#   bash run.sh
#
# FAST REPEAT RUNS: set CACHE_DIR to storage that survives a session restart
# (Google Drive on Colab, an attached dataset / persistent dir on Kaggle). The
# pip wheels (incl. the ~2 GB PaddlePaddle wheel from a slow mirror) and the
# model weights are cached there, so the next cold session skips the big
# downloads instead of installing from zero.
set -e

# ---- GPU preflight --------------------------------------------------------
# PaddleOCR-VL + the LLM extractor are GPU-only. Fail FAST with a clear message
# if no NVIDIA GPU is attached, instead of installing ~2 GB of GPU wheels and
# then dying deep inside paddle with "libcuda.so.1: cannot open shared object
# file" — which only ever means "this runtime has no GPU".
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "!! No NVIDIA GPU detected on this runtime."
  echo "   This service needs a GPU (PaddleOCR-VL + the extractor LLM are GPU-only)."
  echo "   Colab:  Runtime -> Change runtime type -> Hardware accelerator = GPU (T4),"
  echo "           then Reconnect and re-run the cells."
  echo "   Kaggle: Notebook settings -> Accelerator = GPU T4 x2, Internet = On."
  echo "   Verify with:  nvidia-smi   (it must print a GPU table)."
  exit 1
fi

# PaddlePaddle GPU wheel. Default is the CUDA 12.6 index (Kaggle/Colab use CUDA 12.x).
# If it fails, check your CUDA with `nvcc --version` and override, e.g.:
#   PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu118/ bash run.sh
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"

# The PaddlePaddle GPU wheel is ~2 GB from a slow mirror, so pip's default 15s
# read-timeout often aborts it. Use a long timeout and retry the whole install a
# few times (a timed-out partial download is not resumed by pip, so we restart).
PIP_TIMEOUT="${PIP_TIMEOUT:-1000}"

# ---- persistent cache (optional but recommended) --------------------------
if [ -n "$CACHE_DIR" ]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip}"   # cached pip wheels
  export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"                # cached HF model weights (the extractor LLM)
  mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$CACHE_DIR/paddlex"
  # PaddleX / PaddleOCR-VL cache their model weights under ~/.paddlex; redirect
  # that to the persistent cache too (replace a real dir left by an earlier run).
  if [ -e "$HOME/.paddlex" ] && [ ! -L "$HOME/.paddlex" ]; then
    rm -rf "$HOME/.paddlex"
  fi
  ln -sfn "$CACHE_DIR/paddlex" "$HOME/.paddlex"
  echo "==> persistent cache: pip=$PIP_CACHE_DIR  hf=$HF_HOME  paddlex=$CACHE_DIR/paddlex"
fi

# ---- dependencies ---------------------------------------------------------
# On a re-run in the SAME session everything is already installed — skip straight
# to launch. A fresh session reinstalls, but from the cache above (no re-download).
if python3 -c "import paddle, paddleocr, transformers, fastapi" 2>/dev/null; then
  echo "==> dependencies already present — skipping install"
else
  echo "==> installing PaddlePaddle-GPU (large the first time; served from cache afterwards)"
  for attempt in 1 2 3 4 5; do
    python3 -m pip install --timeout "$PIP_TIMEOUT" --retries 10 "paddlepaddle-gpu" -i "$PADDLE_INDEX" && break
    if [ "$attempt" = 5 ]; then
      echo "!! PaddlePaddle install failed after 5 attempts (mirror unreachable)."
      exit 1
    fi
    echo "   attempt $attempt timed out — retrying in 5s ..."
    sleep 5
  done

  echo "==> installing python dependencies"
  python3 -m pip install --timeout "$PIP_TIMEOUT" --retries 10 -q -r requirements.txt

  # The PaddleOCR-VL pipeline needs paddlex's "ocr" extra, matched to the exact
  # paddlex version that paddleocr pulled in above (installing an unpinned
  # paddlex[ocr] could drift the version). Without this the pipeline raises
  # "PaddleOCR-VL-1.6 requires additional dependencies".
  echo "==> installing PaddleOCR-VL pipeline extra (paddlex[ocr])"
  PX_VER="$(python3 -c 'import paddlex,sys; sys.stdout.write(paddlex.__version__)' 2>/dev/null || true)"
  if [ -n "$PX_VER" ]; then
    python3 -m pip install --timeout "$PIP_TIMEOUT" --retries 10 -q "paddlex[ocr]==${PX_VER}"
  else
    python3 -m pip install --timeout "$PIP_TIMEOUT" --retries 10 -q "paddlex[ocr]"
  fi
fi

echo "==> starting server + tunnel"
exec python3 launch.py
