FROM colmap/colmap:latest
LABEL maintainer="Paul-Edouard Sarlin"

# Install Python and git (Ubuntu 24.04 ships with Python 3.12)
# ca-certificates is pre-installed in base image
RUN rm -f /etc/apt/sources.list.d/cuda*.list || true && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements file (code is mounted at runtime)
COPY requirements-hloc.txt /tmp/requirements-hloc.txt

# Global pip config (required because tensorrt-cu12 build calls pip internally)
RUN mkdir -p /root/.config/pip && \
    echo '[global]\nbreak-system-packages = true' > /root/.config/pip/pip.conf

# Install PyTorch with CUDA 12.6 (default pip installs cu130 which requires driver 570+)
# torch 2.6.0 is ~5% faster than 2.13.0 for inference
RUN pip3 install torch==2.6.0+cu126 torchvision==0.21.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126

# Install remaining Python dependencies
# Note: --break-system-packages is set in pip.conf above
# onnxruntime-gpu 1.20.0: ORT 1.28+ requires CUDA 13 libs (libcublasLt.so.13)
# nvidia-cudnn-cu12: let torch manage this dependency (torch 2.6.0 needs 9.5.1.17)
RUN pip3 install \
        -r /tmp/requirements-hloc.txt \
        tensorrt-cu12==10.7.0 \
        tensorrt-cu12-bindings==10.7.0 \
        tensorrt-cu12-libs==10.7.0 \
        onnxruntime-gpu==1.20.0 && \
    rm /tmp/requirements-hloc.txt

# Set library path for cuDNN and TensorRT
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/tensorrt_libs:$LD_LIBRARY_PATH

# Add /app to Python path (code is mounted at runtime, no pip install -e needed)
ENV PYTHONPATH=/app
