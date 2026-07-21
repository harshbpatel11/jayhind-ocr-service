#!/usr/bin/env bash
# One-command setup + run. Works on Kaggle / Colab / any Linux CUDA box.
#   bash run.sh
set -e

# PaddlePaddle GPU wheel. Default is the CUDA 12.6 index (Kaggle/Colab use CUDA 12.x).
# If it fails, check your CUDA with `nvcc --version` and override, e.g.:
#   PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu118/ bash run.sh
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"

echo "==> installing PaddlePaddle-GPU"
python3 -m pip install -q "paddlepaddle-gpu" -i "$PADDLE_INDEX"

echo "==> installing python dependencies"
python3 -m pip install -q -r requirements.txt

echo "==> starting server + tunnel"
exec python3 launch.py
