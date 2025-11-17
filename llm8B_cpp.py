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

#!/usr/bin/env python3
# run_mistral_streaming.py

import subprocess
import os
import time
import psutil
import sys

def print_memory_usage(tag=None):
    """Print current memory usage"""
    mem = psutil.virtual_memory()
    if tag is None:
        tag = ''
    else:
        tag = f" ({tag})"
    g = 1024 **3
    print(f"Memory{tag}: {mem.used/g:.1f}GB / {mem.total/g:.1f}GB ({mem.percent:.1f}%)")
    return mem

def run_with_streaming_output(model_path, prompt):
    """
    Run inference with real-time output streaming
    """
    
    print("=== Running Mistral 7B Q8_0 with GPU Acceleration ===\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Model: {model_path}")
    print(f"Quantization: Q8_0 (8-bit)")
    print(f"Device: M1 Pro GPU (Metal)\n")
    
    mem_before = print_memory_usage(tag='before')
    
    print("="*80)
    print("INFERENCE OUTPUT (streaming)")
    print("="*80 + "\n")
    
    start_time = time.time()

    gpu_layers = 35
    
    # Create the command
    cmd = [
        "llama-cli",
        "--model", model_path,
        "--prompt", prompt,
        "--n-gpu-layers", f"{gpu_layers}", # Offload layers to M1 GPU
        "--ctx-size", "2048",
        "--n-predict", "1024",
        "--temp", "0.7",
        "--top-p", "0.9",
        "--repeat-penalty", "1.1",
        "--threads", "8",  # Use all efficiency cores
        "--batch-size", "512",
        "--no-display-prompt",           # Don't show prompt echo
        "--no-conversation" # skip interactive mode
    ]
    
    # Run with streaming output (NO capture_output!)
    # stdout and stderr go directly to terminal
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Collect output for later analysis
    stdout_lines = []
    stderr_lines = []
    
    # Stream stdout in real-time
    try:
        for line in process.stdout:
            print(line, end='', flush=True)  # Print immediately
            stdout_lines.append(line)
    except KeyboardInterrupt:
        process.kill()
        print("\n\n⚠️  Interrupted by user")
        return None
    
    # Get stderr after stdout is done
    stderr_output = process.stderr.read()
    stderr_lines = stderr_output.split('\n')
    
    # Wait for process to complete
    process.wait()
    
    elapsed = time.time() - start_time

    # Parse performance info from stderr
    for line in stderr_lines:
        if any(keyword in line.lower() for keyword in ['eval time', 'prompt eval', 'llama_perf', 't/s', 'token']):
            print(line)
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    
    print(f"Generation time: {elapsed:.2f}s")
    mem_after = print_memory_usage(tag='after')
    print(f"Memory used: {(mem_after.used - mem_before.used)/1e9:.1f} GiB")
    print(f"GPU layers: {gpu_layers}")
    
    return ''.join(stdout_lines)
    

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
        "--no-display-prompt",           # Don't show prompt echo
        "--no-conversation" # skip interactive mode

    ], capture_output= True, text=True, check=True)
    
    elapsed = time.time() - start_time
    mem_after = psutil.virtual_memory()
    
    print(result.stderr)
    
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
    # response = run_llama_cpp_inference(model_path, prompt)
    response = run_with_streaming_output(model_path, prompt)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
