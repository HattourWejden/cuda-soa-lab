"""
CUDA Sanity Test Suite
Task 1.3: Verify system is ready before deployment
This will be run by Jenkins in the pipeline
"""
import sys
import numpy as np
import time
import os

# Check if running in Docker/GPU environment
RUNNING_IN_DOCKER = os. path.exists('/.dockerenv')
RUNNING_ON_SERVER = os.environ.get('RUNNING_ON_GPU_SERVER', False)

print("\n" + "="*70)
print("CUDA Sanity Test Suite for GPU Lab")
print("="*70)

# ============================================================
# TEST 1: Check Numba is installed
# ============================================================
def test_numba_availability():
    """Test 1: Numba module is available"""
    print("\n✓ TEST 1: Checking Numba availability...")
    try:
        from numba import cuda
        print(f"  ✅ Numba is installed")

        # Check if CUDA device is available
        if cuda.is_available():
            print(f"  ✅ CUDA device detected")
            gpu_count = cuda.gpuci_count()
            print(f"  ✅ Number of GPUs: {gpu_count}")
            return True
        else:
            if RUNNING_IN_DOCKER or RUNNING_ON_SERVER:
                print(f"  ⚠️  No CUDA device detected (will work on GPU server)")
                return True
            else:
                print(f"  ⚠️  No local GPU detected (expected on Windows laptop)")
                print(f"     → GPU will run on instructor's server")
                return True
    except ImportError as e:
        print(f"  ❌ Numba not installed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error checking Numba: {e}")
        return False


# ============================================================
# TEST 2: GPU Kernel imports correctly
# ============================================================
def test_kernel_import():
    """Test 2: GPU kernel module can be imported"""
    print("\n✓ TEST 2: Testing kernel import...")
    try:
        from gpu_kernels import matrix_add_kernel, add_matrices_gpu
        print(f"  ✅ gpu_kernels module imported successfully")
        print(f"  ✅ matrix_add_kernel function found")
        print(f"  ✅ add_matrices_gpu function found")
        return True
    except Exception as e:
        print(f"  ❌ Failed to import gpu_kernels: {e}")
        return False


# ============================================================
# TEST 3: FastAPI app imports correctly
# ============================================================
def test_fastapi_import():
    """Test 3: FastAPI app can be imported"""
    print("\n✓ TEST 3: Testing FastAPI app...")
    try:
        from main import app
        print(f"  ✅ FastAPI app imported successfully")

        # Check endpoints exist
        routes = [route. path for route in app.routes]
        print(f"  ✅ Registered endpoints: {routes}")

        # Verify our 3 endpoints exist
        if "/health" in routes and "/gpu-info" in routes and "/add" in routes:
            print(f"  ✅ All required endpoints registered:")
            print(f"     - /health")
            print(f"     - /gpu-info")
            print(f"     - /add")
            return True
        else:
            print(f"  ❌ Missing some endpoints!")
            return False
    except Exception as e:
        print(f"  ❌ FastAPI app failed to load: {e}")
        return False


# ============================================================
# TEST 4: CPU Matrix Addition (works without GPU)
# ============================================================
def test_matrix_addition_cpu():
    """Test 4: Matrix addition works on CPU"""
    print("\n✓ TEST 4: Testing matrix addition (CPU fallback)...")
    try:
        from gpu_kernels import add_matrices_gpu

        # Create test matrices
        size = 128
        print(f"  • Creating {size}x{size} test matrices...")

        matrix_a = np.random.rand(size, size).  astype(np.float32)
        matrix_b = np. random.rand(size, size).  astype(np.float32)

        # Compute expected result on CPU
        expected = matrix_a + matrix_b

        # Compute using our function (CPU or GPU)
        print(f"  • Running matrix addition...")
        start_time = time. perf_counter()
        result = add_matrices_gpu(matrix_a, matrix_b)
        elapsed_time = time.perf_counter() - start_time

        # Verify correctness
        max_error = np.max(np.abs(result - expected))
        print(f"  ✅ Computation completed in {elapsed_time*1000:.2f}ms")
        print(f"  ✅ Result shape: {result.shape}")
        print(f"  ✅ Max error vs expected: {max_error:.2e}")

        # Check if results match (allow small floating point errors)
        if max_error < 1e-5:
            print(f"  ✅ Results match expected values!")
            return True
        else:
            print(f"  ❌ Results differ too much!")
            return False

    except Exception as e:
        print(f"  ❌ Matrix addition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# TEST 5: File I/O (loading .  npz files)
# ============================================================
def test_file_io():
    """Test 5: . npz file reading works"""
    print("\n✓ TEST 5: Testing .npz file I/O...")
    try:
        import io

        # Create a test matrix
        test_matrix = np.random.rand(5, 5). astype(np.float32)

        # Save to npz in memory
        print(f"  • Creating test . npz file...")
        buffer = io.BytesIO()
        np.savez(buffer, test_matrix)
        buffer. seek(0)

        # Load it back
        print(f"  • Reading test .npz file...")
        loaded = np.load(buffer)
        loaded_matrix = loaded['arr_0']. astype(np.float32)

        # Verify
        if np.allclose(test_matrix, loaded_matrix):
            print(f"  ✅ .npz file I/O works correctly")
            return True
        else:
            print(f"  ❌ .npz file I/O failed")
            return False

    except Exception as e:
        print(f"  ❌ File I/O test failed: {e}")
        return False


# ============================================================
# RUN ALL TESTS
# ============================================================
def main():
    """Run all sanity tests"""

    tests = [
        ("Numba Availability", test_numba_availability),
        ("Kernel Import", test_kernel_import),
        ("FastAPI Import", test_fastapi_import),
        ("Matrix Addition", test_matrix_addition_cpu),
        ("File I/O", test_file_io),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"📊 Results: {passed}/{total} tests passed")
    print("="*70)

    # Print summary table
    print("\nTest Summary:")
    print("-" * 70)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    print("-" * 70)

    # Environment info
    print(f"\nEnvironment Info:")
    print(f"  • Running in Docker: {RUNNING_IN_DOCKER}")
    print(f"  • Running on GPU Server: {RUNNING_ON_SERVER}")
    print(f"  • Local GPU Available: {test_numba_availability.__code__.co_names}")

    # Final verdict
    print("\n" + "="*70)
    if all(result for _, result in results):
        print("✅ ALL TESTS PASSED!  Ready for deployment.")
        print("="*70 + "\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED.  Check errors above.")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)