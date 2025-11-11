FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    curl \
    libgomp1 \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone llama.cpp
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

# Build llama.cpp (optimized for faster build)
WORKDIR /app/llama.cpp
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX2=ON \
        -DGGML_F16C=ON \
        -DGGML_FMA=ON && \
    cmake --build . --config Release --target llama-quantize -j$(nproc)

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN chmod +x scripts/start.sh

# Create directories
RUN mkdir -p /app/logs /app/cache /app/output

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LLAMA_CPP_PATH=/app/llama.cpp \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["./scripts/start.sh"]