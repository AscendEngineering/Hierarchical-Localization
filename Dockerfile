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

# Pre-download models to cache
RUN python3 -c "import torch; torch.hub.load('gmberton/MegaLoc', 'get_trained_model', trust_repo=True)"
