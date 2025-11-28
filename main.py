import time
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from numba import cuda
import io
import subprocess
import logging
import threading
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PROMETHEUS METRICS ====================

# Counters
requests_total = Counter(
    'matrix_addition_requests_total',
    'Total number of matrix addition requests',
    ['status']
)

gpu_operations_total = Counter(
    'gpu_operations_total',
    'Total GPU operations performed'
)

# Histograms
gpu_operation_duration = Histogram(
    'gpu_operation_duration_seconds',
    'GPU operation duration in seconds',
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0)
)

matrix_size_histogram = Histogram(
    'matrix_size_elements',
    'Number of elements in processed matrices',
    buckets=(1000, 10000, 100000, 1000000, 10000000)
)

gpu_memory_used = Histogram(
    'gpu_memory_used_bytes',
    'GPU memory used for operations',
    buckets=(1e6, 10e6, 100e6, 1e9, 10e9)
)

# ==================== CUDA KERNEL DEFINITION ====================

@cuda.jit
def gpu_add_kernel(A, B, C):
    """
    CUDA kernel for matrix addition on GPU.

    How it works:
    1. cuda.grid(2) returns (row, col) - the unique index for each thread
    2. Each thread computes C[row, col] = A[row, col] + B[row, col]
    3. The boundary check ensures threads don't go out of bounds

    Args:
        A: Device array (matrix A)
        B: Device array (matrix B)
        C: Device array (output matrix C = A + B)
    """
    # Get the row and column index of the current thread
    row, col = cuda.grid(2)

    # Boundary check: ensure thread is within matrix bounds
    if row < A. shape[0] and col < A.shape[1]:
        C[row, col] = A[row, col] + B[row, col]


# ==================== FastAPI Application ====================

app = FastAPI(
    title="GPU Matrix Addition Service",
    description="GPU-accelerated matrix addition microservice using Numba CUDA"
)

# Student port (change this to your assigned port)
STUDENT_PORT = 8001
PROMETHEUS_PORT = 8000


# ==================== ENDPOINTS ====================

@app.post("/add")
async def matrix_add(file_a: UploadFile = File(... ), file_b: UploadFile = File(...)):
    """
    POST /add: Performs GPU-accelerated matrix addition.

    Args:
        file_a: NPZ file containing first matrix (as 'arr_0')
        file_b: NPZ file containing second matrix (as 'arr_0')

    Returns:
        JSON with matrix shape, elapsed time, and device info
    """
    try:
        # 1. Read NPZ files from upload
        logger.info("Reading uploaded files...")
        contents_a = await file_a. read()
        contents_b = await file_b.read()

        # 2. Load matrices from NPZ
        matrix_a = np.load(io.BytesIO(contents_a))['arr_0']. astype(np.float32)
        matrix_b = np.load(io.BytesIO(contents_b))['arr_0']. astype(np.float32)

        # 3.  Validate shapes match
        if matrix_a. shape != matrix_b.shape:
            logger.error(f"Shape mismatch: {matrix_a.shape} vs {matrix_b.shape}")
            requests_total.labels(status='error').inc()
            raise HTTPException(
                status_code=400,
                detail=f"Matrix shapes don't match: {matrix_a. shape} vs {matrix_b. shape}"
            )

        logger.info(f"Matrix shape: {matrix_a.shape}")

        # Record matrix size
        total_elements = matrix_a.shape[0] * matrix_a. shape[1]
        matrix_size_histogram.observe(total_elements)

        # 4. Transfer matrices to GPU (device memory)
        logger.info("Transferring matrices to GPU...")
        device_A = cuda.to_device(matrix_a)
        device_B = cuda.to_device(matrix_b)

        # Calculate memory used
        memory_per_matrix = matrix_a.nbytes
        total_memory_used = memory_per_matrix * 3  # A, B, C
        gpu_memory_used.observe(total_memory_used)

        # 5.  Allocate output matrix on GPU
        device_C = cuda.device_array_like(matrix_a)

        # 6. Configure grid and block dimensions
        # Thread block: 32x32 = 1024 threads (optimal for most GPUs)
        threads_per_block = (32, 32)

        # Calculate grid dimensions (rounds up to ensure all elements are covered)
        blocks_x = (matrix_a. shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (matrix_a.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_x, blocks_y)

        logger.info(f"Grid config: blocks={blocks_per_grid}, threads={threads_per_block}")

        # 7. Launch kernel and measure time
        logger.info("Launching GPU kernel...")
        start_time = time.perf_counter()

        gpu_add_kernel[blocks_per_grid, threads_per_block](device_A, device_B, device_C)
        cuda.synchronize()  # Wait for GPU to finish

        elapsed_time = time.perf_counter() - start_time

        # Record metrics
        gpu_operation_duration.observe(elapsed_time)
        gpu_operations_total.inc()

        # 8. Copy result back to host (CPU)
        logger.info("Copying result back to CPU...")
        result = device_C.copy_to_host()

        logger.info(f"GPU computation completed in {elapsed_time:.6f} seconds")
        requests_total.labels(status='success').inc()

        # 9. Return response
        return JSONResponse({
            "matrix_shape": list(matrix_a.shape),
            "elapsed_time": round(elapsed_time, 6),
            "device": "GPU"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during matrix addition: {str(e)}")
        requests_total.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    GET /health: Simple health check endpoint.
    Returns: {"status": "ok"}
    """
    return JSONResponse({"status": "ok"})


@app.get("/gpu-info")
async def gpu_info():
    """
    GET /gpu-info: Returns GPU memory and utilization info.
    Runs nvidia-smi and parses output.

    Returns:
        {
            "gpus": [
                {"gpu": "0", "memory_used_MB": 312, "memory_total_MB": 4096}
            ]
        }
    """
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            raise Exception("nvidia-smi command failed")

        # Parse output
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line. split(',')]
                if len(parts) >= 3:
                    gpus.append({
                        "gpu": parts[0],
                        "memory_used_MB": int(float(parts[1])),
                        "memory_total_MB": int(float(parts[2]))
                    })

        return JSONResponse({"gpus": gpus})

    except Exception as e:
        logger.error(f"Error retrieving GPU info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app. get("/metrics")
async def metrics():
    """
    GET /metrics: Prometheus metrics endpoint
    (Note: Prometheus typically scrapes from http://localhost:8000/metrics)
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn

    # Start Prometheus HTTP server on port 8000 (in a separate thread)
    logger.info(f"Starting Prometheus metrics server on port {PROMETHEUS_PORT}...")
    start_http_server(PROMETHEUS_PORT)

    logger.info(f"Starting GPU Matrix Addition Service on port {STUDENT_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=STUDENT_PORT)