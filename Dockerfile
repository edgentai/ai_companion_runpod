# Use RunPod's Python 3.10 CUDA image
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Copy requirements file
COPY requirements.txt /

# Install dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /requirements.txt

# Copy your handler file
COPY rp_handler.py /

# Set the working directory
WORKDIR /

# RunPod handler command
CMD ["python", "-u", "rp_handler.py"]