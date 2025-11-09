#!/usr/bin/env python3
"""
GGUF Auto-Converter untuk Gensyn RL-Swarm
"""

import os
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, list_repo_commits

# Konfigurasi
REPO_ID = os.environ.get("REPO_ID")
HF_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", 300))
LLAMA_CPP_PATH = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")

# Binaries path
LLAMA_QUANTIZE = f"{LLAMA_CPP_PATH}/build/bin/llama-quantize"
CONVERT_SCRIPT = f"{LLAMA_CPP_PATH}/convert_hf_to_gguf.py"

# Quant types
QUANT_TYPES_STR = os.environ.get("QUANT_TYPES", "F16,Q3_K_M,Q4_K_M,Q5_K_M")
QUANTS = [q.strip() for q in QUANT_TYPES_STR.split(",") if q.strip()]

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log(message, color=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colored_msg = f"{color}{message}{Colors.END}" if color else message
    print(f"[{timestamp}] {colored_msg}", flush=True)

def check_for_updates(api, last_commit_sha):
    """Check update menggunakan commit history"""
    try:
        # Get latest commits
        commits = list_repo_commits(
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN
        )
        
        if not commits:
            return False, last_commit_sha
        
        # Get latest commit SHA
        latest_commit = commits[0]
        current_sha = latest_commit.commit_id
        
        # Check if different from last known commit
        has_update = last_commit_sha is None or current_sha != last_commit_sha
        
        return has_update, current_sha
        
    except Exception as e:
        log(f"⚠️  Error checking updates: {e}", Colors.YELLOW)
        return False, last_commit_sha

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
        "python", CONVERT_SCRIPT,
        local_dir,
        "--outtype", "f16",
        "--outfile", base_gguf
    ]
    
    try:
        result = subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        size_mb = Path(base_gguf).stat().st_size / 1024 / 1024
        log(f"✅ F16 conversion successful ({size_mb:.2f} MB)", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        log(f"❌ Conversion failed: {e.stderr}", Colors.RED)
        return False
    
    # Step 3: Upload
    log("🔨 Step 3/4: Quantizing and Uploading...", Colors.BLUE)
    api = HfApi(token=HF_TOKEN)
    
    # Upload F16
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
    
    # Quantize
    for quant in [q for q in QUANTS if q != "F16"]:
        output_file = f"Qwen3-0.6B-Gensyn-Swarm-{quant}.gguf"
        
        log(f"   🔨 Quantizing to {quant}...", Colors.BLUE)
        quant_cmd = [LLAMA_QUANTIZE, base_gguf, output_file, quant]
        
        try:
            subprocess.run(quant_cmd, check=True, capture_output=True, text=True)
            size_mb = Path(output_file).stat().st_size / 1024 / 1024
            log(f"   ✅ {quant} quantized ({size_mb:.2f} MB)", Colors.GREEN)
            
            log(f"   📤 Uploading {quant}...", Colors.BLUE)
            api.upload_file(
                path_or_fileobj=output_file,
                path_in_repo=output_file,
                repo_id=REPO_ID,
                commit_message=f"🔄 Auto-update: GGUF {quant}"
            )
            log(f"   ✅ {quant} uploaded", Colors.GREEN)
            Path(output_file).unlink()
            
        except Exception as e:
            log(f"   ❌ {quant} failed: {e}", Colors.RED)
    
    # Cleanup
    log("🧹 Step 4/4: Cleaning up...", Colors.BLUE)
    Path(base_gguf).unlink(missing_ok=True)
    subprocess.run(["rm", "-rf", local_dir], check=False)
    
    log("=" * 70, Colors.GREEN)
    log("✨ Conversion completed!", Colors.GREEN)
    log("=" * 70, Colors.GREEN)
    return True

def main():
    if not HF_TOKEN or not REPO_ID:
        log("❌ ERROR: Missing required environment variables!", Colors.RED)
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
    last_commit_sha = None
    idle_count = 0
    
    while True:
        try:
            has_update, current_sha = check_for_updates(api, last_commit_sha)
            
            if has_update:
                idle_count = 0
                log(f"🆕 New commit detected: {current_sha[:8]}", Colors.GREEN)
                
                if convert_and_upload():
                    last_commit_sha = current_sha
                else:
                    log("⚠️  Conversion failed, will retry", Colors.YELLOW)
            else:
                idle_count += 1
                if idle_count == 1:
                    log("💤 No new updates", Colors.YELLOW)
                    log("   ⏸️  Entering IDLE mode...", Colors.YELLOW)
                else:
                    log(f"⏸️  IDLE ({idle_count * CHECK_INTERVAL / 60:.0f} min)", Colors.YELLOW)
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            log("\n🛑 Stopping...", Colors.RED)
            break
        except Exception as e:
            log(f"⚠️  Error: {e}", Colors.RED)
            time.sleep(60)

if __name__ == "__main__":
    main()