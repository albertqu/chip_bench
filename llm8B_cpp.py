#!/usr/bin/env python3
# run_mistral_gguf.py

import subprocess
import os
import time
import psutil

def download_gguf_model():
    """Download pre-quantized GGUF model"""
    
    model_dir = os.path.expanduser("~/llm_models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = f"{model_dir}/mistral-7b-instruct-v0.2.Q8_0.gguf"
    
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return model_path
    
    print("Downloading Mistral 7B Q8_0 (8-bit quantized)...")
    print("Size: ~7.7 GB - this will take a few minutes...\n")
    
    url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf"
    
    subprocess.run([
        "curl", "-L", url,
        "-o", model_path,
        "--progress-bar"
    ], check=True)
    
    print(f"\n✓ Model downloaded: {model_path}")
    return model_path

def run_llama_cpp_inference(model_path, prompt):
    """Run inference with llama.cpp"""
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE WITH LLAMA.CPP (8-bit)")
    print("="*80 + "\n")
    
    mem_before = psutil.virtual_memory()
    start_time = time.time()
    
    # Run llama.cpp with Metal (M1 GPU) acceleration
    result = subprocess.run([
        "llama-cli",
        "--model", model_path,
        "--prompt", prompt,
        "--n-gpu-layers", "35",      # Offload layers to M1 GPU
        "--ctx-size", "2048",         # Context window
        "--temp", "0.7",              # Temperature
        "--top-p", "0.9",             # Nucleus sampling
        "--n-predict", "1024",        # Max tokens to generate
        "--threads", "8",             # Use all efficiency cores
    ], capture_output=True, text=True, check=True)
    
    elapsed = time.time() - start_time
    mem_after = psutil.virtual_memory()
    
    print(result.stdout)
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Generation time: {elapsed:.2f}s")
    print(f"Memory used: {(mem_after.used - mem_before.used)/1e9:.1f} GB")
    print(f"Total memory: {mem_after.used/1e9:.1f} GB / 16 GB")
    print(f"Quantization: Q8_0 (8-bit)")
    
    return result.stdout

def main():
    """Main function"""
    
    print("=== Mistral 7B with 8-bit Quantization (GGUF) ===\n")
    
    # Check if llama.cpp is installed
    try:
        subprocess.run(["llama-cli", "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ llama.cpp not found")
        print("\nInstall with: brew install llama.cpp")
        return
    
    # Download model
    model_path = download_gguf_model()
    
    # The math problem
    prompt = """A rope is 2 meters long and one end is tied to a wall. You make a cut at a uniformly distributed random point along the rope. If the remaining piece that is still attached to the wall is shorter than 1 meter, you stop. If it is longer than 1 meter, you continue cutting the remaining attached part, again at a uniformly distributed random location on what remains.

Question: What is the expected number of cuts needed?

Please solve this step-by-step, showing your mathematical reasoning."""
    
    # Run inference
    response = run_llama_cpp_inference(model_path, prompt)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()