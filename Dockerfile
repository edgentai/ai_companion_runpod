FROM python:3.10-slim

WORKDIR /

# Install dependencies
RUN pip install --no-cache-dir runpod
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
# Copy your handler file
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]