#!/usr/bin/env python3
"""
GGUF Auto-Converter for Gensyn RL-Swarm
Automatically converts HuggingFace model to GGUF format with multiple quantizations
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, snapshot_download, list_repo_commits


# ==================== Configuration ====================
REPO_ID = os.environ.get("REPO_ID")
HF_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", 300))
LLAMA_CPP_PATH = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")

LLAMA_QUANTIZE = f"{LLAMA_CPP_PATH}/build/bin/llama-quantize"
CONVERT_SCRIPT = f"{LLAMA_CPP_PATH}/convert_hf_to_gguf.py"

QUANT_TYPES_STR = os.environ.get("QUANT_TYPES", "F16,Q3_K_M,Q4_K_M,Q5_K_M")
QUANTS = [q.strip() for q in QUANT_TYPES_STR.split(",") if q.strip()]

TEMP_DIR = Path("./temp_model")
BASE_MODEL_NAME = "Qwen3-0.6B-Gensyn-Swarm"


# ==================== ANSI Colors ====================
class Color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


# ==================== Logging ====================
def log(message, color=None, prefix="INFO"):
    """Print timestamped and colored log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colored_prefix = f"{color}[{prefix}]{Color.RESET}" if color else f"[{prefix}]"
    colored_msg = f"{color}{message}{Color.RESET}" if color else message
    print(f"{timestamp} {colored_prefix} {colored_msg}", flush=True)


# ==================== Helper Functions ====================
def cleanup_temp_files(temp_dir, gguf_file=None):
    """Remove temporary files and directories"""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if gguf_file and Path(gguf_file).exists():
            Path(gguf_file).unlink()
    except Exception as e:
        log(f"Cleanup warning: {e}", Color.YELLOW, "WARN")

def run_command(cmd, description):
    """Run subprocess command with error handling"""
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1000
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        log(f"{description} failed", Color.RED, "ERROR")
        # Tampilkan FULL error, bukan hanya 500 chars
        if e.stderr:
            for line in e.stderr.split('\n')[-20:]:  # Last 20 lines
                if line.strip():
                    log(line, Color.RED, "ERROR")
        return False, e.stderr
    except subprocess.TimeoutExpired:
        log(f"{description} timed out", Color.RED, "ERROR")
        return False, "Command timeout"

# ==================== Core Functions ====================
def check_for_updates(api, last_commit_sha):
    """Check for new commits in HuggingFace repository"""
    try:
        commits = list_repo_commits(
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN
        )
        
        if not commits:
            return False, last_commit_sha
        
        latest_commit = commits[0]
        current_sha = latest_commit.commit_id
        has_update = last_commit_sha is None or current_sha != last_commit_sha
        
        return has_update, current_sha
        
    except Exception as e:
        log(f"Update check failed: {e}", Color.YELLOW, "WARN")
        return False, last_commit_sha


def download_model():
    """Download model from HuggingFace with all required files"""
    log("Downloading model from HuggingFace...", Color.BLUE)
    
    try:
        TEMP_DIR.mkdir(exist_ok=True)
        
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(TEMP_DIR),
            ignore_patterns=["*.git*", "*.md", "README"],
            token=HF_TOKEN
        )
        
        # Verify critical files exist
        if not (TEMP_DIR / "model.safetensors").exists():
            log("model.safetensors not found!", Color.RED, "ERROR")
            return False
        
        log("Model downloaded successfully", Color.GREEN)
        return True
        
    except Exception as e:
        log(f"Download failed: {e}", Color.RED, "ERROR")
        return False


def convert_to_f16(output_file):
    """Convert model to F16 GGUF format"""
    log("Converting to F16 GGUF...", Color.BLUE)
    
    cmd = [
        "python", CONVERT_SCRIPT,
        str(TEMP_DIR),
        "--outtype", "f16",
        "--outfile", output_file
    ]
    
    success, output = run_command(cmd, "F16 conversion")
    
    if success and Path(output_file).exists():
        size_mb = Path(output_file).stat().st_size / 1024 / 1024
        log(f"F16 conversion successful ({size_mb:.1f} MB)", Color.GREEN)
        return True
    
    return False


def quantize_model(input_file, output_file, quant_type):
    """Quantize GGUF model to specified quantization type"""
    log(f"Quantizing to {quant_type}...", Color.BLUE)
    
    cmd = [LLAMA_QUANTIZE, input_file, output_file, quant_type]
    success, output = run_command(cmd, f"{quant_type} quantization")
    
    if success and Path(output_file).exists():
        size_mb = Path(output_file).stat().st_size / 1024 / 1024
        log(f"{quant_type} quantized ({size_mb:.1f} MB)", Color.GREEN)
        return True
    
    return False


def upload_to_hf(file_path, commit_msg):
    """Upload file to HuggingFace repository"""
    try:
        api = HfApi(token=HF_TOKEN)
        filename = Path(file_path).name
        
        log(f"Uploading {filename}...", Color.BLUE)
        
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=REPO_ID,
            commit_message=commit_msg
        )
        
        log(f"{filename} uploaded successfully", Color.GREEN)
        return True
        
    except Exception as e:
        log(f"Upload failed for {file_path}: {e}", Color.RED, "ERROR")
        return False


# ==================== Main Conversion Pipeline ====================
def convert_and_upload():
    """Main conversion pipeline: Download -> Convert -> Quantize -> Upload"""
    log("=" * 70, Color.BLUE)
    log("Starting GGUF Conversion Pipeline", Color.BOLD)
    log("=" * 70, Color.BLUE)
    
    base_gguf = f"{BASE_MODEL_NAME}-F16.gguf"
    
    try:
        # Step 1: Download
        if not download_model():
            return False
        
        # Step 2: Convert to F16
        if not convert_to_f16(base_gguf):
            return False
        
        # Step 3: Upload F16
        upload_to_hf(base_gguf, "🔄 Auto-update: GGUF F16")
        
        # Step 4: Quantize and Upload each variant
        for quant in QUANTS:
            if quant == "F16":
                continue  # Already done
            
            output_file = f"{BASE_MODEL_NAME}-{quant}.gguf"
            
            if quantize_model(base_gguf, output_file, quant):
                upload_to_hf(output_file, f"🔄 Auto-update: GGUF {quant}")
                Path(output_file).unlink()  # Cleanup after upload
        
        log("=" * 70, Color.GREEN)
        log("Conversion pipeline completed successfully!", Color.GREEN)
        log("=" * 70, Color.GREEN)
        
        return True
        
    except Exception as e:
        log(f"Conversion pipeline error: {e}", Color.RED, "ERROR")
        return False
        
    finally:
        # Cleanup
        log("Cleaning up temporary files...", Color.BLUE)
        cleanup_temp_files(TEMP_DIR, base_gguf)


# ==================== Main Loop ====================
def main():
    """Main monitoring and conversion loop"""
    
    # Validate environment
    if not HF_TOKEN or not REPO_ID:
        log("Missing required environment variables!", Color.RED, "ERROR")
        log("Required: HUGGINGFACE_ACCESS_TOKEN, REPO_ID", Color.RED, "ERROR")
        sys.exit(1)
    
    # Startup banner
    log("=" * 70, Color.BOLD)
    log("GGUF Auto-Converter for Gensyn RL-Swarm", Color.BOLD + Color.GREEN)
    log("=" * 70, Color.BOLD)
    log(f"Repository: {REPO_ID}", Color.BLUE)
    log(f"Check Interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL/60:.1f} min)", Color.BLUE)
    log(f"Output Formats: {', '.join(QUANTS)}", Color.BLUE)
    log("=" * 70)
    
    api = HfApi(token=HF_TOKEN)
    last_commit_sha = None
    idle_count = 0
    
    # Main monitoring loop
    while True:
        try:
            has_update, current_sha = check_for_updates(api, last_commit_sha)
            
            if has_update:
                idle_count = 0
                log(f"New commit detected: {current_sha[:12]}", Color.GREEN)
                
                if convert_and_upload():
                    last_commit_sha = current_sha
                    log("Waiting for next update...", Color.BLUE)
                else:
                    log("Conversion failed, will retry on next check", Color.YELLOW, "WARN")
            else:
                idle_count += 1
                if idle_count == 1:
                    log("No new updates - entering IDLE mode", Color.YELLOW)
                elif idle_count % 12 == 0:  # Log every hour (12 * 5 min)
                    elapsed_hours = (idle_count * CHECK_INTERVAL) / 3600
                    log(f"IDLE for {elapsed_hours:.1f} hours - waiting for training to resume", Color.YELLOW)
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            log("Shutdown signal received", Color.RED)
            log("Stopping GGUF Auto-Converter... Goodbye!", Color.BLUE)
            break
            
        except Exception as e:
            log(f"Unexpected error: {e}", Color.RED, "ERROR")
            log("Retrying in 60 seconds...", Color.YELLOW, "WARN")
            time.sleep(60)


if __name__ == "__main__":
    main()
