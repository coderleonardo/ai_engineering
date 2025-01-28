print("Importing the libs...")
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gc  # For memory cleanup
print("...finished...")

def load_model():
    """Load model and tokenizer once at startup"""
    local_model_path = "./model"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    return tokenizer, model

def unload_model(model, tokenizer, pipe):
    """Explicitly free memory"""
    del model
    del tokenizer
    del pipe
    gc.collect()  # Force Python garbage collection
    if torch.cuda.is_available():  # Clear GPU cache (safe to call even on CPU-only)
        torch.cuda.empty_cache()

def generate_response(pipe, prompt):
    """Modified to avoid retaining temporary variables"""
    formatted_prompt = f"""<|system|>
    You are a coding assistant for Python, Linux, and ML. Give short answers.</s>
    <|user|>
    {prompt}</s>
    <|assistant|>
    """
    output = pipe(
        formatted_prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    response = output[0]["generated_text"].split("<|assistant|>")[-1].strip()
    
    # Clean intermediate objects
    del formatted_prompt
    del output
    gc.collect()
    
    return response

if __name__ == "__main__":
    try:
        # Load once at startup
        tokenizer, model = load_model()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float32)
        
        print("Ask about coding (type 'exit'):")
        while True:
            user_input = input("> ")
            if user_input.lower() == "exit":
                break
            print(f"\n{generate_response(pipe, user_input)}")
            print(10*"-")
            
    finally:
        # Force cleanup even if user crashes with Ctrl+C
        unload_model(model, tokenizer, pipe)
        print("\nMemory cleaned up. Exiting safely.")