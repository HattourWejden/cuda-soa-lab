"""
GPU Kernel for Matrix Addition using Numba CUDA
Works on GPU servers, falls back to CPU on Windows laptops
"""
from numba import cuda
import numpy as np
import math
import os

# Check if we're in a GPU environment
try:
    GPU_AVAILABLE = cuda.is_available()
except:
    GPU_AVAILABLE = False

print(f"[INFO] GPU Available: {GPU_AVAILABLE}")

# ============================================================
# GPU KERNEL (runs on server with GPU)
# ============================================================
@cuda.jit
def matrix_add_kernel(A, B, C):
    """
    CUDA kernel for 2D matrix addition (GPU version)
    Each thread computes ONE element: C[i,j] = A[i,j] + B[i,j]
    """
    # Get which row and column this thread should compute
    i = cuda.blockIdx. y * cuda.blockDim.y + cuda.threadIdx.y
    j = cuda.blockIdx. x * cuda.blockDim. x + cuda.threadIdx.x

    # Check if this thread's indices are valid
    if i < A.shape[0] and j < A.shape[1]:
        C[i, j] = A[i, j] + B[i, j]


# ============================================================
# MAIN FUNCTION (automatically chooses GPU or CPU)
# ============================================================
def add_matrices_gpu(matrix_a, matrix_b):
    """
    Add two matrices using GPU if available, CPU otherwise.

    This function works in TWO environments:
    1. GPU SERVER (has NVIDIA GPU) → uses GPU kernel
    2.  WINDOWS LAPTOP (no GPU) → uses NumPy fallback

    Args:
        matrix_a: NumPy array (float32)
        matrix_b: NumPy array (float32)

    Returns:
        result: NumPy array with A + B
    """
    # Convert to float32
    matrix_a = np.ascontiguousarray(matrix_a, dtype=np.float32)
    matrix_b = np.ascontiguousarray(matrix_b, dtype=np.float32)

    # ✅ GPU PATH (runs on server)
    if GPU_AVAILABLE:
        print("[GPU] Using CUDA kernel for matrix addition")

        # Send to GPU memory
        d_a = cuda.to_device(matrix_a)
        d_b = cuda.to_device(matrix_b)
        d_c = cuda.device_array_like(d_a)

        # Configure GPU threads
        threads_per_block = (32, 32)  # 32x32 = 1024 threads
        blocks_x = math.ceil(matrix_a.shape[1] / 32)
        blocks_y = math.ceil(matrix_a.shape[0] / 32)

        # Launch kernel
        matrix_add_kernel[(blocks_x, blocks_y), threads_per_block](d_a, d_b, d_c)

        # Get result back
        result = d_c. copy_to_host()
        return result

    # ✅ CPU FALLBACK PATH (runs on Windows laptop)
    else:
        print("[CPU] GPU not available, using NumPy (CPU) fallback")
        result = matrix_a + matrix_b
        return result