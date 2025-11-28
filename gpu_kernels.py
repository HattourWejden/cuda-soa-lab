"""
GPU Kernel for Matrix Addition using Numba CUDA
Works on GPU servers, falls back to CPU
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

# GPU KERNEL (runs on server with GPU)
@cuda.jit
def matrix_add_kernel(A, B, C):

    # Get which row and column this thread should compute
    i = cuda.blockIdx. y * cuda.blockDim.y + cuda.threadIdx.y
    j = cuda.blockIdx. x * cuda.blockDim. x + cuda.threadIdx.x

    # Check if this thread's indices are valid
    if i < A.shape[0] and j < A.shape[1]:
        C[i, j] = A[i, j] + B[i, j]


def add_matrices_gpu(matrix_a, matrix_b):

    matrix_a = np.ascontiguousarray(matrix_a, dtype=np.float32)
    matrix_b = np.ascontiguousarray(matrix_b, dtype=np.float32)
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

    #  CPU FALLBACK PATH (runs on Windows laptop)
    else:
        print("[CPU] GPU not available, using NumPy (CPU) fallback")
        result = matrix_a + matrix_b
        return result