FROM colmap/colmap:latest
MAINTAINER Paul-Edouard Sarlin
ARG PYTHON_VERSION=3.12
# Remove CUDA repo to avoid mirror sync issues (CUDA already installed in base image)
RUN rm -f /etc/apt/sources.list.d/cuda*.list || true
RUN apt-get update -y
RUN apt-get install -y unzip wget software-properties-common git
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python${PYTHON_VERSION}
RUN wget https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py --break-system-packages
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
COPY . /app
WORKDIR app/

RUN pip3 install --break-system-packages --upgrade pip
RUN pip3 install --break-system-packages -r requirements.txt
RUN pip3 install --break-system-packages huggingface_hub safetensors
RUN pip3 install --break-system-packages notebook
RUN pip3 install --break-system-packages -e .

# Install ONNX Runtime with CUDA 12 support for accelerated inference
RUN pip3 install --break-system-packages nvidia-cudnn-cu12

# Install TensorRT for maximum ONNX acceleration (2.5-3x speedup)
# TensorRT requires specific CUDA version - use cu12 for CUDA 12.x
RUN pip3 install --break-system-packages tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs

# Install ONNX Runtime with TensorRT support
RUN pip3 install --break-system-packages onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Set library path for cuDNN and TensorRT
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/tensorrt_libs:$LD_LIBRARY_PATH

# Pre-download models to cache
RUN python3 -c "import torch; torch.hub.load('gmberton/MegaLoc', 'get_trained_model', trust_repo=True)"
