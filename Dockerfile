# Use RunPod's PyTorch image with CUDA support
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables for vLLM
ENV VLLM_USE_CACHE=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Copy requirements file
COPY requirements.txt /

# Install uv and vLLM first using the correct method
RUN python -m pip install --upgrade pip && \
    pip install --upgrade uv && \
    uv pip install vllm==0.10.1 --torch-backend=auto --system

# Install other dependencies from requirements.txt
RUN python -m pip install -r /requirements.txt

# Copy your handler file
COPY rp_handler.py /

# Set the working directory
WORKDIR /

# RunPod handler command
CMD ["python", "-u", "rp_handler.py"]