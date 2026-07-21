#!/usr/bin/env bash
# One-command setup + run. Works on Kaggle / Colab / any Linux CUDA box.
#   bash run.sh
set -e

# PaddlePaddle GPU wheel. Default is the CUDA 12.6 index (Kaggle/Colab use CUDA 12.x).
# If it fails, check your CUDA with `nvcc --version` and override, e.g.:
#   PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu118/ bash run.sh
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"

# The PaddlePaddle GPU wheel is ~2 GB from a slow mirror, so pip's default 15s
# read-timeout often aborts it. Use a long timeout and retry the whole install a
# few times (a timed-out partial download is not resumed by pip, so we restart).
PIP_TIMEOUT="${PIP_TIMEOUT:-1000}"

echo "==> installing PaddlePaddle-GPU (large download; be patient, it retries if the mirror stalls)"
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

echo "==> starting server + tunnel"
exec python3 launch.py
