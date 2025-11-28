"""
CUDA Sanity Test: Verifies GPU and Numba CUDA setup.
Runs a simple GPU vector addition to ensure the environment is correctly configured.
"""

import numpy as np
from numba import cuda
import sys
import logging

logging.basicConfig(level=logging. INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_cuda_available():
    """Test 1: Check if CUDA is available."""
    logger.info("Test 1: Checking CUDA availability...")
    try:
        assert cuda.is_available(), "CUDA is not available!"
        logger.info("✓ CUDA is available")
        return True
    except AssertionError as e:
        logger.error(f"✗ {e}")
        return False


def test_gpu_device():
    """Test 2: Check GPU device."""
    logger.info("Test 2: Checking GPU device...")
    try:
        device = cuda.get_current_device()
        logger.info(f"✓ GPU Device: {device. name. decode() if isinstance(device. name, bytes) else device.name}")
        logger.info(f"  Compute Capability: {device.compute_capability}")
        logger.info(f"  Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
        return True
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False


def test_simple_kernel():
    """Test 3: Run a simple GPU kernel (vector addition)."""
    logger.info("Test 3: Running simple GPU kernel (vector addition)...")

    try:
        # Define a simple kernel
        @cuda.jit
        def simple_add(x, out):
            idx = cuda.grid(1)
            if idx < x.size:
                out[idx] = x[idx] + 1

        # Create test data
        n = 1000
        x = np.arange(n, dtype=np.float32)

        # Transfer to GPU
        d_x = cuda.to_device(x)
        d_out = cuda.device_array(n, dtype=np.float32)

        # Launch kernel (100 threads per block, 10 blocks)
        simple_add[10, 100](d_x, d_out)
        cuda.synchronize()

        # Copy result back
        result = d_out.copy_to_host()

        # Verify result
        expected = x + 1
        assert np.allclose(result, expected), "Kernel result is incorrect!"

        logger. info(f"✓ Kernel executed successfully")
        logger.info(f"  Input range: [{x[0]:. 1f}, {x[-1]:. 1f}]")
        logger.info(f"  Output range: [{result[0]:.1f}, {result[-1]:. 1f}]")
        return True

    except Exception as e:
        logger.error(f"✗ Kernel test failed: {e}")
        return False


def test_matrix_kernel():
    """Test 4: Run a 2D GPU kernel (matrix addition)."""
    logger.info("Test 4: Running 2D GPU kernel (matrix addition)...")

    try:
        @cuda.jit
        def matrix_add(A, B, C):
            row, col = cuda.grid(2)
            if row < A.shape[0] and col < A.shape[1]:
                C[row, col] = A[row, col] + B[row, col]

        # Create test matrices (512x512)
        size = 512
        A = np.ones((size, size), dtype=np.float32)
        B = np.ones((size, size), dtype=np.float32)

        # Transfer to GPU
        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        d_C = cuda.device_array_like(A)

        # Configure grid/block
        threads_per_block = (32, 32)
        blocks_x = (A.shape[0] + 31) // 32
        blocks_y = (A.shape[1] + 31) // 32
        blocks_per_grid = (blocks_x, blocks_y)

        # Launch kernel
        matrix_add[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        cuda.synchronize()

        # Copy result back
        result = d_C. copy_to_host()

        # Verify
        expected = A + B
        assert np.allclose(result, expected), "Matrix kernel result is incorrect!"

        logger.info(f"✓ 2D Matrix kernel executed successfully")
        logger.info(f"  Matrix size: {size}x{size}")
        logger.info(f"  Grid: {blocks_per_grid}, Threads per block: {threads_per_block}")
        return True

    except Exception as e:
        logger.error(f"✗ Matrix kernel test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger. info("CUDA SANITY CHECK")
    logger.info("=" * 60)

    tests = [
        test_cuda_available,
        test_gpu_device,
        test_simple_kernel,
        test_matrix_kernel
    ]

    results = [test() for test in tests]

    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    logger.info("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)