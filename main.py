"""
FastAPI service for GPU-accelerated matrix addition (Task 1.2)
Each student uses a different port: change STUDENT_PORT value
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import time
import subprocess
import uvicorn
import io
from gpu_kernels import add_matrices_gpu

app = FastAPI(
    title="GPU Matrix Addition Service",
    description="Matrix addition microservice with GPU acceleration via Numba CUDA"
)

# ⚠️ TODO: CHANGE THIS PORT TO YOUR ASSIGNED STUDENT PORT
STUDENT_PORT = 8000

@app.get("/health")
async def health_check():
    """
    Health check endpoint (Task 1.2)
    Returns status to verify service is running
    """
    return {"status": "ok"}


@app.get("/gpu-info")
async def get_gpu_info():
    """
    GPU information endpoint (Task 2)
    Queries nvidia-smi to get GPU memory usage

    Returns:
        JSON with list of GPUs and their memory stats

    Example response:
    {
        "gpus": [
            {"gpu": "0", "memory_used_MB": 312, "memory_total_MB": 4096}
        ]
    }
    """
    try:
        # Run nvidia-smi command to get GPU info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            raise HTTPException(status_code=503, detail="nvidia-smi not available")

        # Parse nvidia-smi output
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                gpu_idx, mem_used, mem_total = line.split(', ')
                gpus.append({
                    "gpu": gpu_idx. strip(),
                    "memory_used_MB": int(float(mem_used.strip())),
                    "memory_total_MB": int(float(mem_total.strip()))
                })

        return {"gpus": gpus}

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=503, detail="nvidia-smi timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying GPU: {str(e)}")
# ============================================================
# ENDPOINT 2B: /gpu-load (BONUS)
# ============================================================
@app.get("/gpu-load")
async def get_gpu_load():
    """
    Get GPU utilization percentage.

    Request:
        curl http://localhost:8000/gpu-load

    Response (with GPU):
        {
            "gpu_load": [
                {"gpu": "0", "utilization_percent": 45}
            ]
        }

    Response (no GPU):
        {
            "gpu_load": [],
            "message": "No GPU detected"
        }
    """
    try:
        # Try to run nvidia-smi command
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        # If command failed, GPU not available
        if result.returncode != 0:
            return {
                "gpu_load": [],
                "message": "nvidia-smi not available (GPU not detected)"
            }

        # Parse the output
        gpu_loads = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) == 2:
                    gpu_idx, utilization = parts
                    gpu_loads.append({
                        "gpu": gpu_idx. strip(),
                        "utilization_percent": int(utilization. strip())
                    })

        return {"gpu_load": gpu_loads}

    except subprocess.TimeoutExpired:
        return {"gpu_load": [], "message": "nvidia-smi timeout"}
    except Exception as e:
        return {"gpu_load": [], "message": f"Error: {str(e)}"}

@app.post("/add")
async def add_matrices(
        file_a: UploadFile = File(... , description="First matrix as . npz file"),
        file_b: UploadFile = File(..., description="Second matrix as . npz file")
):
    """
    Matrix addition endpoint (Task 1.2)

    Accepts two . npz files containing NumPy matrices,
    adds them on GPU, and returns result with timing.

    Request:
        curl -X POST "http://localhost:8000/add" \\
             -F "file_a=@matrix1.npz" \\
             -F "file_b=@matrix2.npz"

    Returns:
        {
            "matrix_shape": [512, 512],
            "elapsed_time": 0.0213,
            "device": "GPU"
        }

    Errors:
        400: Matrix shapes don't match
        500: Computation error
    """
    try:
        # Step 1: Read uploaded files into memory
        content_a = await file_a. read()
        content_b = await file_b.read()

        # Step 2: Load NumPy arrays from . npz files
        # . npz files are ZIP archives containing NumPy arrays
        # Default key is 'arr_0' when saved with np.savez()
        npz_a = np.load(io.BytesIO(content_a))
        npz_b = np.load(io.BytesIO(content_b))

        # Extract arrays and convert to float32 (required for GPU kernel)
        matrix_a = npz_a['arr_0'].astype(np.float32)
        matrix_b = npz_b['arr_0'].astype(np. float32)

        # Step 3: Validate that matrices have same shape
        if matrix_a.shape != matrix_b.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Matrix shape mismatch: {matrix_a.shape} vs {matrix_b.shape}"
            )

        # Step 4: Perform GPU addition and measure time
        start_time = time.perf_counter()
        result = add_matrices_gpu(matrix_a, matrix_b)
        elapsed_time = time.perf_counter() - start_time

        # Step 5: Return result with metadata
        return JSONResponse({
            "matrix_shape": list(result.shape),
            "elapsed_time": elapsed_time,
            "device": "GPU"
        })

    except HTTPException:
        # Re-raise HTTPExceptions (validation errors)
        raise
    except Exception as e:
        # Catch any other errors (file format, computation, etc)
        raise HTTPException(
            status_code=500,
            detail=f"Computation error: {str(e)}"
        )


if __name__ == "__main__":
    print(f"🚀 Starting GPU Matrix Addition Service on port {STUDENT_PORT}")
    print(f"📊 Health check: http://localhost:{STUDENT_PORT}/health")
    print(f"📈 GPU info: http://localhost:{STUDENT_PORT}/gpu-info")
    print(f"➕ Matrix addition: http://localhost:{STUDENT_PORT}/add")

    uvicorn.run(app, host="127.0.0.1", port=STUDENT_PORT)