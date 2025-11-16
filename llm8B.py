#!/usr/bin/env python3
# run_mistral_8bit.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import psutil

def print_memory_usage():
    """Print current memory usage"""
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB ({mem.percent:.1f}%)")

def run_mistral_8bit():
    """
    Run Mistral 7B with 8-bit quantization on M1 Pro
    """
    
    print("=== Running Mistral 7B with 8-bit Quantization ===\n")
    
    # Model selection
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Check device
    if torch.backends.mps.is_available():
        print("✓ MPS (M1 GPU) available")
        device = "mps"
    else:
        print("⚠ MPS not available, using CPU")
        device = "cpu"
    
    print(f"Device: {device}\n")
    print_memory_usage()
    
    # Configure 8-bit quantization
    print("\n[1/3] Configuring 8-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,                    # Enable 8-bit quantization
        llm_int8_threshold=6.0,               # Threshold for outlier detection
        llm_int8_has_fp16_weight=False,       # Don't keep FP16 weights
    )
    
    print("  ✓ 8-bit config created")
    print("  ✓ Expected memory: ~7-8 GB (vs ~14 GB in FP16)")
    
    # Load tokenizer
    print("\n[2/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer loaded")
    
    # Load model with 8-bit quantization
    print("\n[3/3] Loading model with 8-bit quantization...")
    print("  (This may take 1-2 minutes on first load)")
    
    start_time = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",              # Automatically map to available device
            low_cpu_mem_usage=True,         # Reduce memory overhead
            torch_dtype=torch.float16,      # Use FP16 for non-quantized parts
        )
    except Exception as e:
        print(f"\n⚠ Error with 8-bit quantization: {e}")
        print("\nNote: bitsandbytes may have limited M1 support.")
        print("Falling back to FP16...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    
    load_time = time.time() - start_time
    print(f"\n✓ Model loaded in {load_time:.1f}s")
    print_memory_usage()
    
    # The math problem
    prompt = """A rope is 2 meters long and one end is tied to a wall. You make a cut at a uniformly distributed random point along the rope. If the remaining piece that is still attached to the wall is shorter than 1 meter, you stop. If it is longer than 1 meter, you continue cutting the remaining attached part, again at a uniformly distributed random location on what remains.

Question: What is the expected number of cuts needed?

Please solve this step-by-step, showing your mathematical reasoning."""
    
    # Format prompt for Mistral Instruct
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80 + "\n")
    
    # Tokenize with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )
    
    # Move to device (for MPS)
    if device == "mps":
        inputs = inputs.to(device)
    
    # Generate response
    print("Generating response...\n")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,           # Allow detailed response
            temperature=0.7,                # Balanced creativity
            top_p=0.9,                      # Nucleus sampling
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response (remove the prompt)
    # Mistral format: [INST] prompt [/INST] response
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response
    
    print("="*80)
    print("LLM RESPONSE")
    print("="*80 + "\n")
    print(response)
    print("\n" + "="*80)
    
    # Performance metrics
    num_generated_tokens = len(outputs[0]) - len(inputs[0])
    tokens_per_second = num_generated_tokens / generation_time
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens generated: {num_generated_tokens}")
    print(f"Speed: {tokens_per_second:.1f} tokens/second")
    print(f"\nQuantization: 8-bit")
    print_memory_usage()
    
    # Memory savings estimate
    print(f"\n💡 Memory savings with 8-bit:")
    print(f"   FP16: ~14 GB")
    print(f"   8-bit: ~7-8 GB")
    print(f"   Savings: ~50% reduction")
    
    return response

if __name__ == "__main__":
    try:
        response = run_mistral_8bit()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install required packages:")
        print("   pip3 install transformers accelerate bitsandbytes --break-system-packages")
        print("2. If bitsandbytes fails on M1, try the alternative script below")