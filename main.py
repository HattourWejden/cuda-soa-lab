"""
FastAPI Service for GPU Matrix Addition with Prometheus Metrics
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import numpy as np
import time
import subprocess
import uvicorn
import io
from gpu_kernels import add_matrices_gpu

app = FastAPI(
    title="GPU Matrix Addition Service",
    description="Add matrices using GPU with Prometheus metrics"
)

STUDENT_PORT = 8000

# 3. PROMETHEUS METRICS
print("[METRICS] Initializing Prometheus metrics...")

# Counter: Total requests
matrix_add_requests_total = Counter(
    'matrix_add_requests_total',
    'Total matrix addition requests',
    ['status']
)

# Histogram: Duration
matrix_add_duration_seconds = Histogram(
    'matrix_add_duration_seconds',
    'Matrix addition duration in seconds',
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
)

# Gauges: GPU metrics
gpu_memory_used_mb = Gauge(
    'gpu_memory_used_mb',
    'GPU memory used in MB'
)

gpu_memory_total_mb = Gauge(
    'gpu_memory_total_mb',
    'Total GPU memory in MB'
)

gpu_utilization_percent = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage'
)

# Counter: Health checks
health_checks_total = Counter(
    'health_checks_total',
    'Total health checks'
)

print("[METRICS] ✓ Prometheus metrics initialized!")

# ENDPOINT: /metrics (Prometheus scrapes this)
@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Request: curl http://localhost:8000/metrics
    """
    print("[METRICS] Serving Prometheus metrics...")
    return Response(generate_latest(REGISTRY), media_type="text/plain")

# ENDPOINT: /health
@app. get("/health")
async def health_check():

    health_checks_total.inc()
    print("[HEALTH] Health check called")
    return {"status": "ok"}
#  ENDPOINT: /gpu-info
@app.get("/gpu-info")
async def get_gpu_info():

    try:
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
            return {
                "gpus": [],
                "message": "nvidia-smi not available"
            }

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) == 3:
                    gpu_idx, mem_used, mem_total = parts
                    mem_used_int = int(float(mem_used. strip()))
                    mem_total_int = int(float(mem_total.strip()))

                    gpus.append({
                        "gpu": gpu_idx. strip(),
                        "memory_used_MB": mem_used_int,
                        "memory_total_MB": mem_total_int
                    })

                    # Update metrics
                    gpu_memory_used_mb.set(mem_used_int)
                    gpu_memory_total_mb. set(mem_total_int)
                    print(f"[GPU-INFO] Updated: {mem_used_int}MB / {mem_total_int}MB")

        return {"gpus": gpus}

    except Exception as e:
        print(f"[GPU-INFO] Error: {e}")
        return {"gpus": [], "message": f"Error: {str(e)}"}

#   ENDPOINT: /gpu-load
@app.get("/gpu-load")
async def get_gpu_load():

    try:
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

        if result.returncode != 0:
            return {
                "gpu_load": [],
                "message": "nvidia-smi not available"
            }

        gpu_loads = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) == 2:
                    gpu_idx, utilization = parts
                    util_int = int(utilization.strip())

                    gpu_loads.append({
                        "gpu": gpu_idx.strip(),
                        "utilization_percent": util_int
                    })

                    # Update metrics
                    gpu_utilization_percent.set(util_int)
                    print(f"[GPU-LOAD] Updated: {util_int}%")

        return {"gpu_load": gpu_loads}

    except Exception as e:
        print(f"[GPU-LOAD] Error: {e}")
        return {"gpu_load": [], "message": f"Error: {str(e)}"}

# 8. ENDPOINT: /add (Matrix Addition)
@app.post("/add")
async def add_matrices(
        file_a: UploadFile = File(... , description="First matrix (. npz)"),
        file_b: UploadFile = File(..., description="Second matrix (.npz)")
):
    try:
        print("\n" + "="*60)
        print("📨 Matrix Addition Request")

        # Read files
        print("📂 Reading files...")
        content_a = await file_a. read()
        content_b = await file_b.read()

        # Load matrices
        print("🔄 Loading matrices...")
        npz_a = np.load(io.BytesIO(content_a))
        npz_b = np.load(io.BytesIO(content_b))

        matrix_a = npz_a['arr_0']. astype(np.float32)
        matrix_b = npz_b['arr_0'].astype(np.float32)

        print(f"✓ Matrix A: {matrix_a.shape}")
        print(f"✓ Matrix B: {matrix_b.shape}")

        # Validate
        if matrix_a.shape != matrix_b.shape:
            print("❌ Shape mismatch!")
            matrix_add_requests_total.labels(status='failure').inc()
            raise HTTPException(
                status_code=400,
                detail=f"Shape mismatch: {matrix_a.shape} vs {matrix_b.shape}"
            )

        # Compute
        print("⚙️  Computing...")
        start_time = time.perf_counter()
        result = add_matrices_gpu(matrix_a, matrix_b)
        elapsed_time = time.perf_counter() - start_time

        print(f"✓ Completed in {elapsed_time*1000:.2f}ms")

        # Update metrics
        matrix_add_requests_total.labels(status='success').inc()
        matrix_add_duration_seconds.observe(elapsed_time)

        response = {
            "matrix_shape": list(result. shape),
            "elapsed_time": elapsed_time,
            "device": "GPU"
        }

        print(f"✅ Success!")
        print("="*60 + "\n")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("="*60 + "\n")
        matrix_add_requests_total.labels(status='failure').inc()
        raise HTTPException(status_code=500, detail=str(e))

# 9. RUN SERVER
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 GPU Matrix Addition Service with Prometheus Metrics")
    print("="*70)
    print(f"📍 URL: http://localhost:{STUDENT_PORT}")
    print(f"\n📋 Endpoints:")
    print(f"   /health          - Health check")
    print(f"   /gpu-info        - GPU memory info")
    print(f"   /gpu-load        - GPU utilization")
    print(f"   /add             - Matrix addition")
    print(f"   /metrics         - Prometheus metrics")
    print(f"\n📚 Docs: http://localhost:{STUDENT_PORT}/docs")
    print("="*70 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=STUDENT_PORT)