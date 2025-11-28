# ============================================================
# Use Ubuntu base image (will use NVIDIA CUDA on GPU server)
# ============================================================
FROM ubuntu:22.04

# Set working directory inside container
WORKDIR /app

# ============================================================
# Install Python and basic tools
# ============================================================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Copy requirements file
# ============================================================
COPY requirements.txt .

# ============================================================
# Install Python dependencies
# ============================================================
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy application code
# ============================================================
COPY main.py .
COPY gpu_kernels.py .
COPY cuda_test.py .

# ============================================================
# Expose ports
# ============================================================
# 8000: FastAPI service
# 8001: Prometheus metrics (future use)
EXPOSE 8000 8001

# ============================================================
# Health check
# ============================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ============================================================
# Run the application
# ============================================================
CMD ["python3", "main.py"]