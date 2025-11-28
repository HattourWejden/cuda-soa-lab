# Use NVIDIA CUDA base image with runtime support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Update package manager and install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3. 10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (optional, but recommended)
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numba \
    numpy \
    scipy \
    python-multipart

# Copy application files
COPY main.py .
COPY cuda_test.py .

# Expose ports
# Student port (change 8001 to your assigned port)
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Set environment variables for better GPU support
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Start the FastAPI service
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]