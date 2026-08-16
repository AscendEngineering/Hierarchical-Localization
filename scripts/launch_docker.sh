#!/bin/bash
# Launch the hloc Docker container with GPU support
#
# Prerequisites:
#   docker build -t hloc:latest .

cd "$(dirname "$0")/.." || exit 1

# Check if ONNX models exist (first-run setup)
SETUP_CMD=""
if [[ ! -f "onnx_models/superpoint_onnx/models/superpoint.onnx" ]]; then
  echo "First run detected — will download/export ONNX models..."
  SETUP_CMD="python3 -m onnx_models.setup && "
fi

docker run --gpus all -it --rm \
  -p 8888:8888 \
  --shm-size=8g \
  -v "$(pwd)":/app \
  -v hloc-torch-cache:/root/.cache/torch \
  -w /app \
  hloc:latest \
  bash -c "${SETUP_CMD}bash"
