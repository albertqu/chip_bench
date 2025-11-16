#!/usr/bin/env python3
# check_mps_memory.py

import torch
import psutil
import subprocess

def check_unified_memory():
    """Check M1 Pro unified memory status"""
    
    print("=== M1 Pro Unified Memory Analysis ===\n")
    
    # Total system memory
    total_memory = psutil.virtual_memory().total / (1024**3)
    available_memory = psutil.virtual_memory().available / (1024**3)
    used_memory = psutil.virtual_memory().used / (1024**3)
    percent_used = psutil.virtual_memory().percent
    
    print(f"Total Unified Memory: {total_memory:.2f} GB")
    print(f"Used Memory: {used_memory:.2f} GB ({percent_used:.1f}%)")
    print(f"Available Memory: {available_memory:.2f} GB")
    print()
    
    # Check MPS (Metal Performance Shaders) availability
    print("--- PyTorch MPS (GPU) Status ---")
    if torch.backends.mps.is_available():
        print("✓ MPS (M1 GPU) is available")
        print("✓ PyTorch can use GPU acceleration")
        
        # MPS doesn't report separate memory like CUDA
        print("\nNote: M1 uses unified memory - no separate GPU memory pool")
        print(f"GPU can use any of the {available_memory:.2f} GB available")
    else:
        print("✗ MPS not available")
        print("  Make sure you have PyTorch 1.12+ with MPS support")
    
    print()
    
    # Estimate usable memory for ML
    print("--- Estimated Usable Memory for ML Workloads ---")
    
    # Account for macOS overhead
    macos_overhead = 3.5  # GB (typical)
    pytorch_overhead = 1.0  # GB (typical)
    
    usable_for_inference = available_memory - 2  # Conservative estimate
    usable_for_training = available_memory - 3    # More conservative
    
    print(f"For Inference: ~{usable_for_inference:.1f} GB")
    print(f"For Training: ~{usable_for_training:.1f} GB")
    print(f"\n(This dynamically shares with CPU as needed)")
    
    return {
        'total': total_memory,
        'available': available_memory,
        'used': used_memory,
        'usable_inference': usable_for_inference,
        'usable_training': usable_for_training
    }

def test_mps_allocation():
    """Test actual MPS memory allocation"""
    
    print("\n=== Testing MPS Memory Allocation ===\n")
    
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping test")
        return
    
    device = torch.device("mps")
    
    try:
        # Allocate progressively larger tensors
        sizes_gb = [0.5, 1.0, 2.0, 4.0, 8.0]
        
        for size_gb in sizes_gb:
            elements = int(size_gb * 1024**3 / 4)  # 4 bytes per float32
            
            print(f"Allocating {size_gb:.1f} GB tensor on MPS...", end=" ")
            
            tensor = torch.randn(elements, dtype=torch.float32, device=device)
            
            # Check current memory
            available = psutil.virtual_memory().available / (1024**3)
            print(f"✓ Success. Available: {available:.2f} GB")
            
            # Clean up
            del tensor
            torch.mps.empty_cache()
            
            if available < 4.0:
                print("\nStopping test - getting low on memory")
                break
                
    except RuntimeError as e:
        print(f"✗ Failed: {e}")

def get_gpu_info():
    """Get M1 Pro GPU core information"""
    
    print("\n=== M1 Pro GPU Information ===\n")
    
    try:
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True,
            text=True
        )
        
        # Extract GPU info
        for line in result.stdout.split('\n'):
            if 'Chipset Model' in line or 'Total Number of Cores' in line or 'Metal' in line:
                print(line.strip())
    except Exception as e:
        print(f"Error getting GPU info: {e}")

if __name__ == "__main__":
    memory_info = check_unified_memory()
    get_gpu_info()
    test_mps_allocation()
    