FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev git wget curl \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1-mesa-glx \
    libgtk-3-0 libgtk-3-dev libnss3 libxss1 libasound2 xauth xvfb x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy only YOLOv5 and DeepSort directories that exist
COPY yolov5/ /workspace/yolov5/
COPY deep_sort-master/ /workspace/deep_sort-master/
COPY yolo_deepsort_integration.py /workspace/
COPY requirements.txt /workspace/

# Install YOLOv5 requirements
RUN pip3 install -r /workspace/yolov5/requirements.txt

# Install DeepSort requirements
RUN pip3 install -r /workspace/deep_sort-master/requirements-gpu.txt

# Install additional required packages
RUN pip3 install scikit-learn filterpy

WORKDIR /workspace
RUN mkdir -p /workspace/results

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | CUDA: {torch.version.cuda}')"

CMD ["/bin/bash"]
