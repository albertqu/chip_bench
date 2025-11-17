#!/usr/bin/env python3
# mistral_mps_attempt.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil

def attempt_mps_loading():
    """
    Attempt various strategies to load Mistral 7B on MPS
    """
    
    print("=== Attempting to Load Mistral 7B on MPS ===\n")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return None
    
    print("✓ MPS is available\n")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    mem_before = psutil.virtual_memory()
    print(f"Memory before: {mem_before.used/1e9:.1f} GB / 16 GB\n")
    
    # Load tokenizer
    print("[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded\n")
    
    # Strategy 1: Try loading with auto device map
    print("[2/3] Attempting to load model with auto device mapping...")
    print("      Splitting between MPS and CPU...\n")
    
    try:
        start_time = time.time()
        
        # Try with automatic device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",           # Auto-split across devices
            max_memory={
                0: "10GB",               # Limit MPS to 10GB
                "cpu": "6GB"             # Rest on CPU
            },
            low_cpu_mem_usage=True,
            offload_folder="./offload",  # Offload to disk if needed
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f}s")
        print(f"✓ Device map: {model.hf_device_map}\n")
        
        mem_after = psutil.virtual_memory()
        print(f"Memory after: {mem_after.used/1e9:.1f} GB / 16 GB\n")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Auto device mapping failed: {e}\n")
        return None, tokenizer

def attempt_mps_manual_split():
    """
    Manually split model between MPS and CPU
    """
    
    print("=== Attempting Manual Device Split ===\n")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # Load on CPU first
        print("Loading model on CPU first...\n")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        
        print("✓ Model loaded on CPU\n")
        print("Attempting to move embedding layer to MPS...\n")
        
        # Try to move just the embedding layer to MPS
        try:
            model.model.embed_tokens = model.model.embed_tokens.to('mps')
            print("✓ Embeddings on MPS")
        except Exception as e:
            print(f"⚠️  Could not move embeddings to MPS: {e}")
        
        # Try moving first few layers to MPS
        print("\nAttempting to move first 10 layers to MPS...\n")
        
        num_layers_on_gpu = 0
        for i in range(10):
            try:
                model.model.layers[i] = model.model.layers[i].to('mps')
                num_layers_on_gpu += 1
                print(f"✓ Layer {i} on MPS")
            except Exception as e:
                print(f"❌ Layer {i} failed: {e}")
                break
        
        print(f"\n✓ Successfully moved {num_layers_on_gpu} layers to MPS")
        print(f"✓ Remaining {32 - num_layers_on_gpu} layers on CPU\n")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Manual split failed: {e}")
        return None, None

def attempt_bf16_mps():
    """
    Try with BF16 instead of FP16
    """
    
    print("=== Attempting with BF16 ===\n")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        print("Loading with BF16 on MPS...\n")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Try BF16 instead
            device_map="mps",
            low_cpu_mem_usage=True,
        )
        
        print("✓ Model loaded with BF16 on MPS!\n")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ BF16 loading failed: {e}\n")
        return None, None

def run_inference(model, tokenizer, device_info="mixed"):
    """
    Run inference with the loaded model
    """
    
    if model is None:
        print("❌ No model loaded, cannot run inference")
        return
    
    prompt = """A rope is 2 meters long and one end is tied to a wall. You make a cut at a uniformly distributed random point along the rope. If the remaining piece that is still attached to the wall is shorter than 1 meter, you stop. If it is longer than 1 meter, you continue cutting the remaining attached part, again at a uniformly distributed random location on what remains.

Question: What is the expected number of cuts needed?

Please solve this step-by-step, showing your mathematical reasoning."""
    
    messages = [{"role": "user", "content": prompt}]
    
    print("="*80)
    print(f"GENERATING RESPONSE (Device: {device_info})")
    print("="*80 + "\n")
    
    try:
        # Tokenize
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        # Move inputs to appropriate device
        # If model has device_map, inputs should go to first device
        if hasattr(model, 'hf_device_map'):
            first_device = list(model.hf_device_map.values())[0]
            inputs = inputs.to(first_device)
        else:
            inputs = inputs.to('cpu')  # Safe default
        
        print("Generating...\n")
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
        print(f"Tokens: {num_tokens}")
        print(f"Speed: {tokens_per_second:.1f} tokens/second")
        print(f"Memory: {psutil.virtual_memory().used/1e9:.1f} GB / 16 GB")
        
        return response
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("EXPERIMENTAL: Attempting to load Mistral 7B on M1 MPS\n")
    print("⚠️  Warning: This will likely fail due to MPS limitations\n")
    print("="*80 + "\n")
    
    # Try Strategy 1: Auto device mapping
    print("STRATEGY 1: Auto Device Mapping\n")
    model, tokenizer = attempt_mps_loading()
    
    if model is not None:
        print("\n✅ SUCCESS with auto device mapping!\n")
        run_inference(model, tokenizer, "auto (MPS + CPU)")
        return
    
    # Try Strategy 2: Manual split
    print("\n" + "="*80 + "\n")
    print("STRATEGY 2: Manual Layer Split\n")
    model, tokenizer = attempt_mps_manual_split()
    
    if model is not None:
        print("\n✅ SUCCESS with manual split!\n")
        run_inference(model, tokenizer, "manual (partial MPS)")
        return
    
    # Try Strategy 3: BF16
    print("\n" + "="*80 + "\n")
    print("STRATEGY 3: BFloat16\n")
    model, tokenizer = attempt_bf16_mps()
    
    if model is not None:
        print("\n✅ SUCCESS with BF16!\n")
        run_inference(model, tokenizer, "MPS (BF16)")
        return
    
    # All strategies failed
    print("\n" + "="*80)
    print("ALL STRATEGIES FAILED")
    print("="*80 + "\n")
    print("❌ Unable to load Mistral 7B on MPS\n")
    print("This is expected due to MPS's 4GB tensor size limit.\n")
    print("Recommended alternatives:")
    print("  1. Use Ollama: ollama run mistral:7b")
    print("  2. Use GGUF with llama.cpp")
    print("  3. Use a smaller model (Phi-3 Mini, Qwen 2.5 3B)")
    print("  4. Run on CPU with FP16 (slow but works)")

if __name__ == "__main__":
    main()