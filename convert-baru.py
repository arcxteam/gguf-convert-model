#!/usr/bin/env python3
"""
GGUF Auto-Converter for all HuggingFace model 
to GGUF format with multiple quantizations
"""

import os
import sys
import time
import subprocess
import shutil
import threading
import contextlib
import io
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

# Suppress HuggingFace telemetry
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# ==================== ANSI Colors ====================
class Color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'


# ==================== Logging ====================
def log(message, color=None, prefix="INFO"):
    """Print timestamped and colored log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colored_prefix = f"{color}[{prefix}]{Color.RESET}" if color else f"[{prefix}]"
    colored_msg = f"{color}{message}{Color.RESET}" if color else message
    print(f"{timestamp} {colored_prefix} {colored_msg}", flush=True)


# ==================== Progress Tracker ====================
class ProgressTracker:
    """Track and display progress at key milestones only"""
    
    def __init__(self, operation_name, file_size_mb=None):
        self.operation_name = operation_name
        self.file_size_mb = file_size_mb
        self.reported_milestones = set()
        self.start_time = time.time()
        
    def report_milestone(self, percentage):
        """Report progress at milestone percentages"""
        milestone = (percentage // 25) * 25
        
        if milestone not in self.reported_milestones and milestone > 0:
            self.reported_milestones.add(milestone)
            elapsed = time.time() - self.start_time
            
            size_info = f" ({self.file_size_mb:.1f} MB)" if self.file_size_mb else ""
            speed_info = f" in {elapsed:.1f}s" if milestone == 100 else ""
            
            log(
                f"{self.operation_name}: {milestone}%{size_info}{speed_info}",
                Color.CYAN
            )


# ==================== Helper Functions ====================
@contextlib.contextmanager
def suppress_hf_output():
    """Completely suppress HuggingFace progress bars and output"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


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
            timeout=3600
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        log(f"{description} failed", Color.RED, "ERROR")
        if e.stderr:
            for line in e.stderr.split('\n')[-20:]:
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
    """Download model and tokenizer files with clean progress"""
    log("Downloading model from HuggingFace...", Color.BLUE)
    
    try:
        TEMP_DIR.mkdir(exist_ok=True)
        
        tracker = ProgressTracker("Download")
        download_done = threading.Event()
        
        def track_download():
            for milestone in [25, 50, 75]:
                if download_done.is_set():
                    break
                time.sleep(4)
                tracker.report_milestone(milestone)
        
        thread = threading.Thread(target=track_download, daemon=True)
        thread.start()
        
        with suppress_hf_output():
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(TEMP_DIR),
                allow_patterns=["*.safetensors", "*.json"],
                token=HF_TOKEN
            )
            
            if not (TEMP_DIR / "model.safetensors").exists():
                download_done.set()
                log("model.safetensors not found!", Color.RED, "ERROR")
                return False
            
            snapshot_download(
                repo_id="Qwen/Qwen3-0.6B",
                local_dir=str(TEMP_DIR),
                allow_patterns=[
                    "vocab.json",
                    "merges.txt",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json"
                ],
                token=HF_TOKEN
            )
        
        download_done.set()
        tracker.report_milestone(100)
        log("✓ Model and tokenizer downloaded", Color.GREEN)
        return True
        
    except Exception as e:
        if 'download_done' in locals():
            download_done.set()
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
    
    tracker = ProgressTracker("Conversion")
    
    def track_conversion():
        for milestone in [25, 50, 75]:
            time.sleep(4)
            tracker.report_milestone(milestone)
    
    thread = threading.Thread(target=track_conversion, daemon=True)
    thread.start()
    
    success, output = run_command(cmd, "F16 conversion")
    
    if success and Path(output_file).exists():
        size_mb = Path(output_file).stat().st_size / 1024 / 1024
        tracker.file_size_mb = size_mb
        tracker.report_milestone(100)
        log(f"✓ F16 GGUF created ({size_mb:.1f} MB)", Color.GREEN)
        return True
    
    return False


def quantize_model(input_file, output_file, quant_type):
    """Quantize GGUF model"""
    log(f"Quantizing to {quant_type}...", Color.BLUE)
    
    cmd = [LLAMA_QUANTIZE, input_file, output_file, quant_type]
    
    tracker = ProgressTracker(f"Quantize {quant_type}")
    
    def track_quantize():
        for milestone in [25, 50, 75]:
            time.sleep(2)
            tracker.report_milestone(milestone)
    
    thread = threading.Thread(target=track_quantize, daemon=True)
    thread.start()
    
    success, output = run_command(cmd, f"{quant_type} quantization")
    
    if success and Path(output_file).exists():
        size_mb = Path(output_file).stat().st_size / 1024 / 1024
        tracker.file_size_mb = size_mb
        tracker.report_milestone(100)
        log(f"✓ {quant_type} created ({size_mb:.1f} MB)", Color.GREEN)
        return True
    
    return False


def upload_to_hf(file_path, commit_msg):
    """Upload file to HuggingFace with clean progress tracking"""
    try:
        api = HfApi(token=HF_TOKEN)
        filename = Path(file_path).name
        file_size_mb = Path(file_path).stat().st_size / 1024 / 1024
        
        log(f"Uploading {filename} ({file_size_mb:.1f} MB)...", Color.BLUE)
        
        tracker = ProgressTracker(f"Upload {filename}", file_size_mb)
        upload_done = threading.Event()
        
        def track_upload():
            estimated_time = file_size_mb / 30
            start = time.time()
            
            while not upload_done.is_set():
                elapsed = time.time() - start
                if elapsed >= estimated_time * 0.25 and 25 not in tracker.reported_milestones:
                    tracker.report_milestone(25)
                elif elapsed >= estimated_time * 0.50 and 50 not in tracker.reported_milestones:
                    tracker.report_milestone(50)
                elif elapsed >= estimated_time * 0.75 and 75 not in tracker.reported_milestones:
                    tracker.report_milestone(75)
                time.sleep(1)
        
        thread = threading.Thread(target=track_upload, daemon=True)
        thread.start()
        
        with suppress_hf_output():
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=REPO_ID,
                commit_message=commit_msg
            )
        
        upload_done.set()
        tracker.report_milestone(100)
        log(f"✓ {filename} uploaded", Color.GREEN)
        return True
        
    except Exception as e:
        if 'upload_done' in locals():
            upload_done.set()
        log(f"Upload failed: {e}", Color.RED, "ERROR")
        return False


# ==================== Main Conversion Pipeline ====================
def convert_and_upload():
    """Main conversion pipeline"""
    log("=" * 70, Color.BLUE)
    log("Starting GGUF Conversion Pipeline", Color.BOLD)
    log("=" * 70, Color.BLUE)
    
    base_gguf = f"{BASE_MODEL_NAME}-F16.gguf"
    
    try:
        if not download_model():
            return False
        
        if not convert_to_f16(base_gguf):
            return False
        
        upload_to_hf(base_gguf, "Auto-update: GGUF F16")
        
        for quant in QUANTS:
            if quant == "F16":
                continue
            
            output_file = f"{BASE_MODEL_NAME}-{quant}.gguf"
            
            if quantize_model(base_gguf, output_file, quant):
                upload_to_hf(output_file, f"Auto-update: GGUF {quant}")
                Path(output_file).unlink()
        
        log("=" * 70, Color.GREEN)
        log("✓ Conversion pipeline completed successfully!", Color.GREEN)
        log("=" * 70, Color.GREEN)
        
        return True
        
    except Exception as e:
        log(f"Pipeline error: {e}", Color.RED, "ERROR")
        return False
        
    finally:
        log("Cleaning up temporary files...", Color.BLUE)
        cleanup_temp_files(TEMP_DIR, base_gguf)


# ==================== Main Loop ====================
def main():
    """Main monitoring and conversion loop"""
    
    if not HF_TOKEN or not REPO_ID:
        log("Missing required environment variables!", Color.RED, "ERROR")
        log("Required: HUGGINGFACE_ACCESS_TOKEN, REPO_ID", Color.RED, "ERROR")
        sys.exit(1)
    
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
    
    while True:
        try:
            has_update, current_sha = check_for_updates(api, last_commit_sha)
            
            if has_update:
                idle_count = 0
                log(f"🆕 New commit detected: {current_sha[:12]}", Color.GREEN)
                
                if convert_and_upload():
                    last_commit_sha = current_sha
                    log("⏳ Waiting for next update...", Color.BLUE)
                else:
                    log("Conversion failed, will retry on next check", Color.YELLOW, "WARN")
            else:
                idle_count += 1
                if idle_count == 1:
                    log("💤 No new updates - entering IDLE mode", Color.YELLOW)
                elif idle_count % 12 == 0:
                    elapsed_hours = (idle_count * CHECK_INTERVAL) / 3600
                    log(f"⏸️  IDLE for {elapsed_hours:.1f}h - waiting for training", Color.YELLOW)
            
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