# Use RunPod's Python 3.10 CUDA image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04


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