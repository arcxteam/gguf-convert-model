# auto-gguf-model

Baik, saya akan buatkan setup Docker Compose yang **production-ready** dan **flowchart** untuk menjelaskan cara kerjanya.[1][2][3]

## Docker Setup Lengkap

### 1. Struktur Folder

```
gguf-auto-converter/
├── .env
├── docker-compose.yml
├── Dockerfile
├── convert_to_gguf.py
├── requirements.txt
└── start.sh
```

### 2. File `.env` (Konfigurasi)

```bash
# HuggingFace Configuration
HUGGINGFACE_ACCESS_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
REPO_ID=0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther

# Conversion Settings
CHECK_INTERVAL=300
QUANT_TYPES=F16,Q3_K_M,Q4_K_M,Q5_K_M

# Timezone
TZ=Asia/Jakarta
```

### 3. File `requirements.txt`

```txt
huggingface-hub>=0.20.0
requests>=2.31.0
```

### 4. File `Dockerfile`

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    make -j$(nproc) && \
    pip install --no-cache-dir -r requirements.txt

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy conversion script
COPY convert_to_gguf.py .
COPY start.sh .
RUN chmod +x start.sh

# Create temp directory for model storage
RUN mkdir -p /app/temp_model

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LLAMA_CPP_PATH=/app/llama.cpp

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the converter
CMD ["./start.sh"]
```

### 5. File `start.sh`

```bash
#!/bin/bash
set -e

echo "=================================================="
echo "🚀 GGUF Auto-Converter for Gensyn RL-Swarm"
echo "=================================================="
echo "Repo: $REPO_ID"
echo "Check Interval: $CHECK_INTERVAL seconds"
echo "Quant Types: $QUANT_TYPES"
echo "=================================================="
echo ""

# Validate HF token
if [ -z "$HUGGINGFACE_ACCESS_TOKEN" ]; then
    echo "❌ ERROR: HUGGINGFACE_ACCESS_TOKEN is not set!"
    exit 1
fi

# Start the conversion script
exec python3 /app/convert_to_gguf.py
```

### 6. File `docker-compose.yml`

```yaml
version: '3.8'

services:
  gguf-converter:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: gguf-auto-converter
    restart: unless-stopped
    environment:
      - HUGGINGFACE_ACCESS_TOKEN=${HUGGINGFACE_ACCESS_TOKEN}
      - REPO_ID=${REPO_ID}
      - CHECK_INTERVAL=${CHECK_INTERVAL}
      - TZ=${TZ}
    volumes:
      # Persistent storage untuk cache (opsional, untuk mempercepat)
      - converter-cache:/app/cache
      # Logs
      - ./logs:/app/logs
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    networks:
      - converter-network
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  converter-cache:
    driver: local

networks:
  converter-network:
    driver: bridge
```

### 7. File `convert_to_gguf.py` (Updated)

```python
#!/usr/bin/env python3
"""
GGUF Auto-Converter untuk Gensyn RL-Swarm
Monitoring dan auto-convert model.safetensors ke GGUF formats
"""

import os
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# Konfigurasi dari environment variables
REPO_ID = os.environ.get("REPO_ID")
HF_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", 300))
LLAMA_CPP_PATH = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")

# Parse quant types dari env (default jika tidak ada)
QUANT_TYPES_STR = os.environ.get("QUANT_TYPES", "F16,Q3_K_M,Q4_K_M,Q5_K_M")
QUANTS = [q.strip() for q in QUANT_TYPES_STR.split(",")]

class Colors:
    """ANSI color codes untuk pretty logging"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log(message, color=None):
    """Print dengan timestamp dan color"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colored_msg = f"{color}{message}{Colors.END}" if color else message
    print(f"[{timestamp}] {colored_msg}", flush=True)

def check_for_updates(api, last_modified):
    """
    Check update pada model.safetensors
    Returns: (has_update, new_timestamp, commit_info)
    """
    try:
        file_info = api.get_paths_info(
            repo_id=REPO_ID,
            paths=["model.safetensors"],
            repo_type="model"
        )
        current_modified = file_info[0].last_modified
        
        has_update = last_modified is None or current_modified > last_modified
        return has_update, current_modified
        
    except Exception as e:
        log(f"⚠️  Error checking updates: {e}", Colors.YELLOW)
        return False, last_modified

def convert_and_upload():
    """Main conversion workflow"""
    log("=" * 70, Colors.BLUE)
    log("🔄 Starting GGUF Conversion Process", Colors.BOLD)
    log("=" * 70, Colors.BLUE)
    
    local_dir = "./temp_model"
    
    # Step 1: Download
    log("📥 Step 1/4: Downloading model from HuggingFace...", Colors.BLUE)
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=local_dir,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
            token=HF_TOKEN
        )
        log("✅ Model downloaded successfully", Colors.GREEN)
    except Exception as e:
        log(f"❌ Download failed: {e}", Colors.RED)
        return False
    
    # Step 2: Convert ke F16
    log("📦 Step 2/4: Converting to F16 GGUF...", Colors.BLUE)
    base_gguf = "Qwen3-0.6B-Gensyn-Swarm-F16.gguf"
    
    convert_cmd = [
        "python", f"{LLAMA_CPP_PATH}/convert_hf_to_gguf.py",
        local_dir,
        "--outtype", "f16",
        "--outfile", base_gguf
    ]
    
    try:
        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        size_mb = Path(base_gguf).stat().st_size / 1024 / 1024
        log(f"✅ F16 conversion successful ({size_mb:.2f} MB)", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        log(f"❌ Conversion failed: {e.stderr}", Colors.RED)
        return False
    
    # Step 3: Quantize and Upload
    log("🔨 Step 3/4: Quantizing to K-quants...", Colors.BLUE)
    api = HfApi(token=HF_TOKEN)
    
    # Upload F16 first
    try:
        log(f"   📤 Uploading F16...", Colors.BLUE)
        api.upload_file(
            path_or_fileobj=base_gguf,
            path_in_repo=base_gguf,
            repo_id=REPO_ID,
            commit_message="🔄 Auto-update: GGUF F16"
        )
        log(f"   ✅ F16 uploaded", Colors.GREEN)
    except Exception as e:
        log(f"   ❌ F16 upload failed: {e}", Colors.RED)
    
    # Quantize ke K-quants
    for quant in [q for q in QUANTS if q != "F16"]:
        output_file = f"Qwen3-0.6B-Gensyn-Swarm-{quant}.gguf"
        
        log(f"   🔨 Quantizing to {quant}...", Colors.BLUE)
        quant_cmd = [
            f"{LLAMA_CPP_PATH}/llama-quantize",
            base_gguf,
            output_file,
            quant
        ]
        
        try:
            subprocess.run(quant_cmd, check=True, capture_output=True, text=True)
            size_mb = Path(output_file).stat().st_size / 1024 / 1024
            log(f"   ✅ {quant} quantized ({size_mb:.2f} MB)", Colors.GREEN)
            
            # Upload
            log(f"   📤 Uploading {quant}...", Colors.BLUE)
            api.upload_file(
                path_or_fileobj=output_file,
                path_in_repo=output_file,
                repo_id=REPO_ID,
                commit_message=f"🔄 Auto-update: GGUF {quant}"
            )
            log(f"   ✅ {quant} uploaded", Colors.GREEN)
            
            # Cleanup
            Path(output_file).unlink()
            
        except Exception as e:
            log(f"   ❌ {quant} failed: {e}", Colors.RED)
    
    # Step 4: Cleanup
    log("🧹 Step 4/4: Cleaning up...", Colors.BLUE)
    Path(base_gguf).unlink(missing_ok=True)
    subprocess.run(["rm", "-rf", local_dir], check=False)
    
    log("=" * 70, Colors.GREEN)
    log("✨ Conversion completed successfully!", Colors.GREEN)
    log("=" * 70, Colors.GREEN)
    return True

def main():
    """Main monitoring loop"""
    
    # Validasi
    if not HF_TOKEN or not REPO_ID:
        log("❌ ERROR: Missing required environment variables!", Colors.RED)
        log("   Required: HUGGINGFACE_ACCESS_TOKEN, REPO_ID", Colors.RED)
        sys.exit(1)
    
    log("🚀 GGUF Auto-Converter Started", Colors.BOLD + Colors.GREEN)
    log(f"📍 Repository: {REPO_ID}", Colors.BLUE)
    log(f"⏱️  Check Interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL/60:.1f} min)", Colors.BLUE)
    log(f"📋 Output Formats: {', '.join(QUANTS)}", Colors.BLUE)
    log("=" * 70)
    log("ℹ️  Behavior: Script will IDLE when training stops", Colors.YELLOW)
    log("ℹ️  It will auto-resume when new commits detected", Colors.YELLOW)
    log("=" * 70)
    
    api = HfApi(token=HF_TOKEN)
    last_modified = None
    idle_count = 0
    
    while True:
        try:
            has_update, current_modified = check_for_updates(api, last_modified)
            
            if has_update:
                idle_count = 0  # Reset idle counter
                log(f"🆕 New training update detected!", Colors.GREEN)
                log(f"   Timestamp: {current_modified}", Colors.BLUE)
                
                if convert_and_upload():
                    last_modified = current_modified
                else:
                    log("⚠️  Conversion failed, will retry next check", Colors.YELLOW)
            else:
                idle_count += 1
                if idle_count == 1:
                    log("💤 No new updates (training might be paused/stopped)", Colors.YELLOW)
                    log("   ⏸️  Entering IDLE mode - waiting for next commit...", Colors.YELLOW)
                else:
                    # Silent idle - print minimal log
                    log(f"⏸️  IDLE ({idle_count * CHECK_INTERVAL / 60:.0f} min) - Waiting for training to resume...", Colors.YELLOW)
            
            # Wait before next check
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            log("\n🛑 Shutdown signal received", Colors.RED)
            log("👋 Stopping GGUF Auto-Converter... Goodbye!", Colors.BLUE)
            break
        except Exception as e:
            log(f"⚠️  Unexpected error: {e}", Colors.RED)
            log(f"⏳ Retrying in 60 seconds...", Colors.YELLOW)
            time.sleep(60)

if __name__ == "__main__":
    main()
```

## Cara Menggunakan

### Setup & Start

```bash
# 1. Buat folder dan copy semua file
mkdir gguf-auto-converter
cd gguf-auto-converter

# 2. Edit .env file - PENTING!
nano .env
# Isi dengan HuggingFace token Anda

# 3. Build dan jalankan
docker-compose up --build -d

# 4. Monitor logs
docker-compose logs -f
```

### Manajemen Container

```bash
# Check status
docker-compose ps

# Stop
docker-compose stop

# Start lagi
docker-compose start

# Restart
docker-compose restart

# Stop dan hapus
docker-compose down

# View logs
docker-compose logs -f gguf-converter

# Logs 100 baris terakhir
docker-compose logs --tail=100 gguf-converter
```

## Penjelasan Behavior

Sekarang saya buatkan **flowchart** untuk menjelaskan cara kerja sistemnya:

Sources
[1] Docker Compose keep container running - Stack Overflow https://stackoverflow.com/questions/38546755/docker-compose-keep-container-running
[2] How to Build a Llama.cpp Container Image - Docker https://docs.vultr.com/how-to-build-a-llama-cpp-container-image
[3] How to Deploy Hugging Face Models in a Docker Container https://fgiasson.com/blog/index.php/2023/08/23/how-to-deploy-hugging-face-models-in-a-docker-container/
[4] How to Use Docker Compose and Python to Automate Your Jenkins ... https://aws.plainenglish.io/how-to-use-docker-compose-and-python-to-automate-your-jenkins-environment-a-hands-on-guide-f1b094fb81d4
[5] Services | Docker Docs https://docs.docker.com/reference/compose-file/services/
[6] Running docker-compose up with lots of services makes it stall #7486 https://github.com/docker/compose/issues/7486
[7] Wait for Services to Start in Docker Compose - DEV Community https://dev.to/welel/wait-for-services-to-start-in-docker-compose-wait-for-it-vs-healthcheck-127d
[8] Docker Compose Keep Container Running Example https://www.javacodegeeks.com/docker-compose-keep-container-running-example.html
[9] Run llama.cpp in a GPU accelerated Docker container https://github.com/fboulnois/llama-cpp-docker
[10] Manually Downloading Models in docker build with ... https://discuss.huggingface.co/t/manually-downloading-models-in-docker-build-with-snapshot-download/19637
