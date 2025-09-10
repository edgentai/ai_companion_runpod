import os
import runpod
from vllm import LLM, SamplingParams
from huggingface_hub import login
from datetime import datetime

# Get token from environment variable
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("HuggingFace authentication successful!")
else:
    print("Warning: No HF_TOKEN found, proceeding without authentication")

# Initialize the model outside the handler for better performance
# This way the model is loaded once when the worker starts, not on every request
print("Loading model...")
llm = LLM(
    model="aimagic/jinx-gpt-oss-20b-vllm-compatible",  # HuggingFace model path
    dtype="bfloat16",
    trust_remote_code=True,  # Important for modified models
    async_scheduling=True,
    gpu_memory_utilization=0.9
    )
print("Model loaded successfully!")

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
    
    # System message with added reasoning constraint
    system_content = f"""{system_prompt}
Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: low

# Valid channels: final. Channel must be included for every message. Never use analysis or commentry channels. That will break the entire system."""
    
    prompt_parts.append(f"<|start|>system<|message|>{system_content}<|end|>")
    
    # Add developer message (instructions)
    developer_message = "# Instructions\n\nYou are an intelligent assistant which gives answers. Do not reason in more than 50 words."
    prompt_parts.append(f"<|start|>developer<|message|>{developer_message}<|end|>")
    
    # Add chat history
    for message in chat_history:
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if role == 'system':
            # Skip system messages in history as we already have one
            continue
        elif role == 'user':
            prompt_parts.append(f"<|start|>user<|message|>{content}<|end|>")
        elif role == 'assistant':
            # For assistant messages, include the channel tag
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
    input_data = event['input']
    
    # Get user inputs
    system_prompt = input_data.get('system_prompt', 'You are Jinx, a creative and intelligent assistant.')
    chat_history = input_data.get('chat_history', [])  # Expecting list of message dicts
    new_message = input_data.get('new_message', '')
    
    # Get generation parameters (with defaults)
    max_tokens = input_data.get('max_tokens', 512)
    temperature = input_data.get('temperature', 0.75)
    top_p = input_data.get('top_p', 0.9)
    
    # Validate new_message
    if not new_message:
        return {
            "status": "error",
            "error": "new_message is required"
        }
    
    # Build the prompt using the template format
    prompt = build_prompt(system_prompt, chat_history, new_message)
    
    print(f"Processing chat completion")
    print(f"System prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"System prompt: {system_prompt}")
    print(f"Chat history length: {len(chat_history)} messages")
    print(f"New message: {new_message[:100]}..." if len(new_message) > 100 else f"New message: {new_message}")
    print(f"Generation params: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    
    try:
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
            #stop=["<|end|>", "<|start|>"]  # Stop tokens for this model format
        )
        
        # Generate completion
        outputs = llm.generate([prompt], sampling_params)
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        
        print("Chat completion generated successfully")
        print(f"Generated text length: {len(generated_text)} characters")
        
        # Build response in a format similar to the original
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": outputs[0].outputs[0].finish_reason,
                "index": 0
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),  # Approximate token count
                "completion_tokens": len(generated_text.split()),  # Approximate token count
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        }
        
        # Return the response
        return {
            "status": "success",
            "response": response
        }
        
    except Exception as e:
        print(f"Error generating completion: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})