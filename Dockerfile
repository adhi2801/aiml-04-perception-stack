# CPU image for the offline parts: demo on a recorded drive, evaluation scripts, lint and unit tests.
# CARLA, TensorRT and the ROS2 nodes are NOT in this image (CARLA renders on the host GPU; ROS2 has its own
# ros:jazzy CI job). Latency measured in this container is CPU latency, not the GPU numbers in the README.
#
#   docker build -t perception-stack:cpu .
#   docker run --rm -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
#       perception-stack:cpu make demo PY=python
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends make libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU torch wheels; everything else as in requirements.txt minus the GPU-only / simulator packages
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision|onnxruntime-gpu|tensorrt|carla|wandb)' requirements.txt > req-cpu.txt \
    && pip install --no-cache-dir -r req-cpu.txt onnxruntime ruff==0.14.5

COPY . .
RUN pip install --no-cache-dir --no-deps -e .
ENV CUDA_VISIBLE_DEVICES=""
CMD ["make", "check", "PY=python"]
