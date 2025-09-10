# import runpod
# import time  

# def handler(event):
# #   This function processes incoming requests to your Serverless endpoint.
# #
# #    Args:
# #        event (dict): Contains the input data and request metadata
# #       
# #    Returns:
# #       Any: The result to be returned to the client
    
#     # Extract input data
#     print(f"Worker Start")
#     input = event['input']
    
#     prompt = input.get('prompt')  
#     seconds = input.get('seconds', 0)  

#     print(f"Received prompt: {prompt}")
#     print(f"Sleeping for {seconds} seconds...")
    
#     # You can replace this sleep call with your own Python code
#     time.sleep(seconds)  
    
#     return prompt 

# # Start the Serverless function when the script is run
# if __name__ == '__main__':
#     runpod.serverless.start({'handler': handler })
import os
import runpod
from llama_cpp import Llama
from huggingface_hub import login

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
llm = Llama.from_pretrained(
    repo_id="Jinx-org/Jinx-gpt-oss-20b-GGUF",
    filename="jinx-gpt-oss-20b-Q2_K.gguf"
)
print("Model loaded successfully!")

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
    system_prompt = input_data.get('system_prompt', 'You are a helpful assistant.')
    chat_history = input_data.get('chat_history', [])  # Expecting list of message dicts
    new_message = input_data.get('new_message', '')
    system_prompt += ". Reasoning: low"
    
    # Validate new_message
    if not new_message:
        return {
            "status": "error",
            "error": "new_message is required"
        }
    
    # Build messages array
    messages = []
    
    # Add system prompt
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add chat history
    if chat_history:
        messages.extend(chat_history)
    
    # Add the new user message
    messages.append({
        "role": "user",
        "content": new_message
    })
    
    # Hardcoded generation parameters
    max_tokens = 512
    temperature = 0.7
    top_p = 0.95
    
    print(f"Processing chat completion with {len(messages)} messages")
    print(f"System prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"System prompt: {system_prompt}")
    print(f"Chat history length: {len(chat_history)} messages")
    print(f"New message: {new_message[:100]}..." if len(new_message) > 100 else f"New message: {new_message}")
    
    try:
        # Create chat completion
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        print("Chat completion generated successfully")
        
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