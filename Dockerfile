FROM ubuntu:22.04
# Set working directory inside container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY gpu_kernels.py .
COPY cuda_test.py .

# 8000: FastAPI service
# 8001: Prometheus metrics (future use)
EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "main.py"]