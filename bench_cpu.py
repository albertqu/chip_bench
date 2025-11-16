# ml_benchmark.py
import time
import numpy as np

def cpu_benchmark():
    """Test CPU performance with matrix operations"""
    print("Testing CPU Performance...")
    size = 50000
    start = time.time()
    
    # Large matrix multiplication
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    c = np.dot(a, b)
    
    duration = time.time() - start
    print(f"CPU Matrix Multiplication ({size}x{size}): {duration:.2f}s")
    
    # M1 should complete this in 5-8 seconds typically
    if duration < 10:
        print("✓ CPU Performance: GOOD")
    else:
        print("✗ CPU Performance: SLOW")

def memory_benchmark():
    """Test memory bandwidth"""
    print("\nTesting Memory Bandwidth...")
    size = 1000000000  # 1GB
    start = time.time()
    
    arr = np.random.rand(size)
    result = np.sum(arr)
    
    duration = time.time() - start
    bandwidth = (size * 8) / (duration * 1e9)  # GB/s
    print(f"Memory Bandwidth: {bandwidth:.2f} GB/s")
    
    if bandwidth > 50:
        print("✓ Memory: EXCELLENT")
    elif bandwidth > 30:
        print("✓ Memory: GOOD")
    else:
        print("✗ Memory: SLOW")

if __name__ == "__main__":
    cpu_benchmark()
    memory_benchmark()