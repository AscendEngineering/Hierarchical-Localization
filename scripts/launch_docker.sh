#!/bin/bash
# Launch the hloc Docker container with TensorRT support

cd "$(dirname "$0")/.." || exit 1

# LD_LIBRARY_PATH includes:
# - TensorRT libs (for ONNX Runtime TensorRT EP)
# - cuDNN libs (for CUDA operations)
docker run --gpus all -it --rm -p 8888:8888 \
  --shm-size=8g \
  -v "$(pwd)":/app \
  -w /app \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt_libs:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib \
  hloc:latest \
  bash
