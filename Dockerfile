FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    libgomp1 \
    libcurl4-openssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp with CMake
WORKDIR /app/llama.cpp
RUN mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release -j$(nproc)

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# bacl to cd /app & copy script
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY convert_to_gguf.py .
COPY start.sh .
RUN chmod +x start.sh

# Create temp directory
RUN mkdir -p /app/temp_model /app/logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV LLAMA_CPP_PATH=/app/llama.cpp

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["./start.sh"]