#!/usr/bin/env python3
# run_mistral_8bit_alternative.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil

def print_memory_usage():
    """Print current memory usage"""
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB ({mem.percent:.1f}%)")

def quantize_model_8bit(model):
    """
    Apply dynamic 8-bit quantization to model
    Works better on M1 than bitsandbytes
    """
    print("Applying dynamic 8-bit quantization...")
    
    # Quantize linear layers to int8
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize linear layers
        dtype=torch.qint8    # 8-bit integers
    )
    
    return quantized_model

def run_mistral_quantized():
    """
    Run Mistral 7B with PyTorch's built-in quantization
    Better M1 support than bitsandbytes
    """
    
    print("=== Running Mistral 7B with 8-bit Quantization (PyTorch Native) ===\n")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ MPS (M1 GPU) available\n")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU\n")
    
    print_memory_usage()
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in FP32 first (for quantization)
    print("\n[2/3] Loading model...")
    start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Load in FP32 first
        low_cpu_mem_usage=True
    )
    
    load_time = time.time() - start_time
    print(f"  ✓ Model loaded in {load_time:.1f}s")
    print_memory_usage()
    
    # Quantize to 8-bit
    print("\n[3/3] Quantizing to 8-bit...")
    model = quantize_model_8bit(model)
    
    # Note: Quantized models run on CPU for now
    # MPS support for quantized models is limited
    model = model.to('cpu')
    
    print("  ✓ Model quantized")
    print_memory_usage()
    
    # The math problem
    prompt = """A rope is 2 meters long and one end is tied to a wall. You make a cut at a uniformly distributed random point along the rope. If the remaining piece that is still attached to the wall is shorter than 1 meter, you stop. If it is longer than 1 meter, you continue cutting the remaining attached part, again at a uniformly distributed random location on what remains.

Question: What is the expected number of cuts needed?

Please solve this step-by-step, showing your mathematical reasoning."""
    
    messages = [{"role": "user", "content": prompt}]
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80 + "\n")
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )
    
    # Generate
    print("Generating response (this may take longer on CPU)...\n")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response
    
    print("="*80)
    print("LLM RESPONSE")
    print("="*80 + "\n")
    print(response)
    print("\n" + "="*80)
    
    # Metrics
    num_tokens = len(outputs[0]) - len(inputs[0])
    tokens_per_second = num_tokens / generation_time
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens generated: {num_tokens}")
    print(f"Speed: {tokens_per_second:.1f} tokens/second")
    print(f"Quantization: 8-bit (PyTorch dynamic)")
    print_memory_usage()
    
    return response

if __name__ == "__main__":
    response = run_mistral_quantized()