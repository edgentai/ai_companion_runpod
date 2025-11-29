import os
import runpod
import boto3
from vllm import LLM, SamplingParams
from huggingface_hub import login
from faster_whisper import WhisperModel, BatchedInferencePipeline
from datetime import datetime

# HF auth
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("HuggingFace authentication successful!")

# AWS clients
print("Initializing AWS clients...")
try:
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.environ.get("AWS_REGION", "ap-south-1")
    
    if aws_key and aws_secret:
        sqs = boto3.client("sqs", aws_access_key_id=aws_key, aws_secret_access_key=aws_secret, region_name=aws_region)
        s3 = boto3.client("s3", aws_access_key_id=aws_key, aws_secret_access_key=aws_secret, region_name=aws_region)
        print("AWS clients initialized successfully!")
    else:
        sqs = None
        s3 = None
except Exception as e:
    print(f"Warning: Failed to initialize AWS clients: {e}")
    sqs = None
    s3 = None

# TEST 1: Initialize vLLM FIRST (your original working order)
print("TEST 1: Loading vLLM FIRST...")
llm = LLM(
    model="aimagic/jinx-gpt-oss-20b-vllm-compatible",
    dtype="bfloat16",
    trust_remote_code=True,
    async_scheduling=True,  # Keep this - it worked before
    enforce_eager=True,
    gpu_memory_utilization=0.85,  # Reduced to leave room for Whisper
    max_num_seqs=256,
    max_model_len=4096,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    disable_custom_all_reduce=True,
    swap_space=4,
    block_size=32,
)
print("vLLM loaded successfully!")

# NOW initialize Whisper AFTER vLLM
print("TEST 1: Loading Whisper AFTER vLLM...")
try:
    whisper_model = WhisperModel("small", device="cuda", compute_type="int8")
    batched_whisper = BatchedInferencePipeline(model=whisper_model)
    print("Whisper model initialized successfully!")
except Exception as e:
    print(f"Warning: Failed to initialize Whisper model: {e}")
    batched_whisper = None

def handler(event):
    """Test handler - just check if both models loaded"""
    return {
        "status": "success",
        "test": "TEST 1 - vLLM first, then Whisper",
        "vllm_loaded": llm is not None,
        "whisper_loaded": batched_whisper is not None,
        "message": "If you see this, both models loaded successfully with async_scheduling=True!"
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})