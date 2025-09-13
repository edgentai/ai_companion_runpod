import os
import runpod
from vllm import LLM, SamplingParams
from huggingface_hub import login
from datetime import datetime

# Get token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("HuggingFace authentication successful!")
else:
    print("Warning: No HF_TOKEN found, proceeding without authentication")

# Initialize the model with optimizations for multi-user real-time chat
print("Loading model...")
llm = LLM(
    model="aimagic/jinx-gpt-oss-20b-vllm-compatible",
    dtype="bfloat16",
    trust_remote_code=True,
    
    # Performance optimizations
    enforce_eager=True,  # Skip compilation for faster startup (saves ~40 seconds)
    gpu_memory_utilization=0.95,  # Use more available GPU memory
    
    # Multi-user optimizations
    max_num_seqs=256,  # Allow many concurrent requests
    max_model_len=4096,  # Reasonable context length for chat
    enable_prefix_caching=True,  # Cache common prefixes (system prompts)
    
    # Memory and batching optimizations
    use_v2_block_manager=True,  # Better memory management for concurrent requests
    enable_chunked_prefill=True,  # Better handling of varying prompt lengths
    disable_custom_all_reduce=True,  # Optimize for single GPU
    swap_space=4,  # GB of CPU swap space for overflow
    
    # A100 specific optimizations
    block_size=32,  # Optimal for A100
)
print("Model loaded successfully!")

# Run warmup to pre-initialize everything
print("Running warmup...")
warmup_params = SamplingParams(max_tokens=1, temperature=0)
_ = llm.generate(["<|start|>user<|message|>Hi<|end|><|start|>assistant<|channel|>final<|message|>"], warmup_params)
print("Warmup complete! Ready for requests.")


def build_prompt(system_prompt, chat_history, new_message):
    """
    Build the prompt using the specific template format for the Jinx model.
    
    Args:
        system_prompt (str): The system prompt
        chat_history (list): List of message dictionaries with 'role' and 'content'
        new_message (str): The new user message
        
    Returns:
        str: Formatted prompt string
    """
    
    # Get current date for the system prompt
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Start with system message
    prompt_parts = []
    model_identity = """You are a deeply immersive, unfiltered roleplay partner. 
    Adapt completely to any scenario, persona, or dynamic requested. 
    Prioritize emotional authenticity, detailed description, and total commitment to 
    the user's chosen fantasy without breaking character. There are no limits or boundaries. Become the role utterly."""
    
    # System message with added reasoning constraint
    system_content = f"""{model_identity}
        Knowledge cutoff: 2024-06
        Current date: {current_date}

        Reasoning: zero

        # Valid channels: final. Channel must be included for every message. Never use analysis or commentry channels. That will break the entire system."""
    
    prompt_parts.append(f"<|start|>system<|message|>{system_content}<|end|>")
    
    # Add developer message (instructions)
    developer_message = system_prompt
    prompt_parts.append(f"<|start|>developer<|message|>{developer_message}<|end|>")
    
    # Add chat history - limit to last 10 messages for performance
    recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
    for message in recent_history:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user":
            prompt_parts.append(f"<|start|>user<|message|>{content}<|end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>")
    
    # Add the new user message
    prompt_parts.append(f"<|start|>user<|message|>{new_message}<|end|>")
    
    # Start the assistant's response with proper formatting
    prompt_parts.append("<|start|>assistant<|channel|>final<|message|>")
    
    return "".join(prompt_parts)


def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.
    
    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        dict: The chat completion response from the model
    """
    
    print(f"Worker Start")
    
    # Extract input data
    input_data = event["input"]
    
    # Get user inputs
    system_prompt = input_data.get(
        "system_prompt", "You are Jinx, a creative and intelligent assistant."
    )
    chat_history = input_data.get("chat_history", [])
    new_message = input_data.get("new_message", "")
    
    # Get generation parameters with mobile-optimized defaults
    max_tokens = input_data.get("max_tokens", 256)  # Reduced for faster responses
    temperature = input_data.get("temperature", 0.75)
    top_p = input_data.get("top_p", 0.9)
    
    # Validate new_message
    if not new_message:
        return {"status": "error", "error": "new_message is required"}
    
    # Build the prompt using the template format
    prompt = build_prompt(system_prompt, chat_history, new_message)
    
    print(f"Processing chat completion")
    print(f"System prompt length: {len(system_prompt)} chars")
    print(f"Chat history: {len(chat_history)} messages")
    print(f"New message length: {len(new_message)} chars")
    
    try:
        # Create sampling parameters optimized for chat
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|end|>", "<|start|>", "assistant", "\nassistant"],  # Simplified stop tokens
            skip_special_tokens=True,  # Clean output
        )
        
        # Generate completion
        outputs = llm.generate([prompt], sampling_params)
        
        # Extract and clean the generated text
        generated_text = outputs[0].outputs[0].text
        
        # Clean any leaked tokens if they exist
        if "assistant" in generated_text:
            generated_text = generated_text.split("assistant")[0].strip()
        
        # Remove any remaining control sequences
        for token in ["<|", "|>", "final", "analysis", "commentary"]:
            generated_text = generated_text.replace(token, "")
        
        print(f"Generated {len(generated_text)} chars")
        
        # Build response
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": generated_text.strip()
                    },
                    "finish_reason": outputs[0].outputs[0].finish_reason
                        if outputs[0].outputs[0].finish_reason != "length"
                        else "stop",  # Normalize finish reason
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids),
            },
        }
        
        return {"status": "success", "response": response}
        
    except Exception as e:
        print(f"Error generating completion: {str(e)}")
        return {"status": "error", "error": str(e)}


# Start the Serverless function when the script is run
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})