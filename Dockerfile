# Use RunPod's PyTorch image with CUDA support
# FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
# Set environment variables for vLLM and A100 optimization
ENV VLLM_USE_CACHE=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install build dependencies
RUN apt-get update && apt-get install -y \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /

# Install vLLM with FlashAttention 3
RUN python -m pip install --upgrade pip && \
    pip install vllm==0.10.1 && \
    pip install -r /requirements.txt

# Copy your handler file
COPY rp_handler.py /

# Set the working directory
WORKDIR /

# RunPod handler command
CMD ["python", "-u", "rp_handler.py"]