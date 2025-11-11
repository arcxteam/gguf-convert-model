import sys
import io
import time
import subprocess
import shutil
import threading
import contextlib
from pathlib import Path
from datetime import datetime, timedelta
from huggingface_hub import HfApi, snapshot_download, list_repo_commits

from src.utils.logger import Logger, ProgressTracker

log = Logger()

def suppress_output():
    """Context manager to suppress stdout/stderr"""
    @contextlib.contextmanager
    def _suppress():
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    return _suppress()


def run_command(cmd, description, timeout=3600):
    """Run subprocess command with error handling"""
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        log.error(f"{description} failed")
        if e.stderr:
            for line in e.stderr.split('\n')[-15:]:
                if line.strip():
                    log.error(f"  {line}")
        return False, e.stderr
    except subprocess.TimeoutExpired:
        log.error(f"{description} timed out")
        return False, "Timeout"


def cleanup_temp(temp_dir, *files):
    """Remove temporary directory and files"""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        for file in files:
            if file and Path(file).exists():
                Path(file).unlink()
    except Exception as e:
        log.warning(f"Cleanup warning: {e}")


def cleanup_old_local_files(output_dir, max_age_hours=24):
    """Delete local GGUF files older than specified hours"""
    try:
        if not output_dir.exists():
            return
        
        now = datetime.now()
        cutoff = now - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        for file in output_dir.glob("*.gguf"):
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            if file_time < cutoff:
                file_size_mb = file.stat().st_size / 1024 / 1024
                file.unlink()
                deleted_count += 1
                log.info(f"Deleted old file: {file.name} ({file_size_mb:.1f} MB, age: {(now - file_time).total_seconds() / 3600:.1f}h)")
        
        if deleted_count > 0:
            log.success(f"Cleaned up {deleted_count} old files from {output_dir}")
    
    except Exception as e:
        log.warning(f"Local cleanup warning: {e}")


def check_for_updates(config, last_sha):
    """Check for new commits in HuggingFace repo"""
    try:
        api = HfApi(token=config.hf_token)
        commits = list_repo_commits(
            repo_id=config.repo_id,
            repo_type="model",
            token=config.hf_token
        )
        
        if not commits:
            return False, last_sha
        
        latest_sha = commits[0].commit_id
        has_update = last_sha is None or latest_sha != last_sha
        return has_update, latest_sha
    
    except Exception as e:
        log.warning(f"Update check failed: {e}")
        return False, last_sha


def download_model(config):
    """Download model with tokenizer fallback"""
    log.info("Downloading model from HuggingFace...")
    
    try:
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        
        tracker = ProgressTracker("Download")
        tracker.start_timer()
        done = threading.Event()
        
        def track():
            for m in [25, 50, 75]:
                if done.is_set():
                    break
                time.sleep(3)
                tracker.report(m)
        
        thread = threading.Thread(target=track, daemon=True)
        thread.start()
        
        # Download model files
        with suppress_output():
            snapshot_download(
                repo_id=config.repo_id,
                local_dir=str(config.temp_dir),
                allow_patterns=["*.safetensors", "*.bin", "*.json", "tokenizer*", "*.model"],
                token=config.hf_token
            )
        
        # Check model weights
        if not any((config.temp_dir / f).exists() for f in ["model.safetensors", "pytorch_model.bin"]):
            done.set()
            log.error("No model weights found!")
            return False
        
        # Download base tokenizer if specified
        if config.base_model_tokenizer:
            log.info(f"Downloading tokenizer from {config.base_model_tokenizer}...")
            with suppress_output():
                snapshot_download(
                    repo_id=config.base_model_tokenizer,
                    local_dir=str(config.temp_dir),
                    allow_patterns=["vocab*", "merges.txt", "tokenizer*", "special_tokens*"],
                    token=config.hf_token
                )
        
        done.set()
        tracker.report(100)
        return True
    
    except Exception as e:
        if 'done' in locals():
            done.set()
        log.error(f"Download failed: {e}")
        return False


def convert_to_f16(config, output_file):
    """Convert model to F16 GGUF format"""
    log.info("Converting to F16 GGUF...")
    
    cmd = [
        "python", str(config.convert_script),
        str(config.temp_dir),
        "--outtype", "f16",
        "--outfile", output_file
    ]
    
    tracker = ProgressTracker("Conversion")
    tracker.start_timer()
    
    def track():
        for m in [25, 50, 75]:
            time.sleep(4)
            tracker.report(m)
    
    thread = threading.Thread(target=track, daemon=True)
    thread.start()
    
    success, _ = run_command(cmd, "F16 conversion", config.conversion_timeout)
    
    if success and Path(output_file).exists():
        size_mb = Path(output_file).stat().st_size / 1024 / 1024
        tracker.file_size_mb = size_mb
        tracker.report(100)
        return True
    
    return False

def quantize_model(config, input_file, output_file, quant_type):
    """Quantize GGUF to specified type"""
    log.info(f"Quantizing to {quant_type}...")
    
    cmd = [str(config.quantize_binary), input_file, output_file, quant_type]
    
    tracker = ProgressTracker(f"Quantize {quant_type}")
    tracker.start_timer()
    
    def track():
        for m in [25, 50, 75]:
            time.sleep(2)
            tracker.report(m)
    
    thread = threading.Thread(target=track, daemon=True)
    thread.start()
    
    success, _ = run_command(cmd, f"{quant_type} quantization", config.conversion_timeout)
    
    if success and Path(output_file).exists():
        size_mb = Path(output_file).stat().st_size / 1024 / 1024
        tracker.file_size_mb = size_mb
        tracker.report(100)
        return True
    
    return False

def upload_to_hf(config, file_path, commit_msg):
    """Upload file with flexible modes: same_repo, new_repo, local_only"""
    
    filename = Path(file_path).name
    file_size_mb = Path(file_path).stat().st_size / 1024 / 1024
    
    # Mode 1: Local only
    if config.upload_mode == "local_only":
        dest = config.output_dir / filename
        shutil.copy(file_path, dest)
        log.success(f"Saved locally: {dest} ({file_size_mb:.1f} MB)")
        
        # Auto-cleanup old files
        cleanup_old_local_files(config.output_dir, config.local_cleanup_hours)
        return True
    
    # Mode 2 & 3: Upload to HuggingFace
    try:
        api = HfApi(token=config.hf_token)
        
        # Determine target repo
        if config.upload_mode == "new_repo":
            target = config.target_repo
            
            # Check/create repo
            try:
                api.repo_info(repo_id=target, repo_type="model")
                log.info(f"Target repo exists: {target}")
            except Exception:
                log.warning(f"Creating new repo: {target}")
                try:
                    api.create_repo(
                        repo_id=target,
                        repo_type="model",
                        private=False,
                        exist_ok=True
                    )
                    log.success(f"Repo created: {target}")
                except Exception as e:
                    log.error(f"Failed to create repo: {e}")
                    return False
        else:
            target = config.repo_id
        
        log.info(f"Uploading {filename} ({file_size_mb:.1f} MB) to {target}...")
        
        tracker = ProgressTracker(f"Upload {filename}", file_size_mb)
        tracker.start_timer()
        done = threading.Event()
        
        def track():
            est_time = file_size_mb / 30
            start = time.time()
            while not done.is_set():
                elapsed = time.time() - start
                for m in [25, 50, 75]:
                    if elapsed >= est_time * (m / 100) and m not in tracker.milestones:
                        tracker.report(m)
                time.sleep(1)
        
        thread = threading.Thread(target=track, daemon=True)
        thread.start()
        
        with suppress_output():
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=target,
                commit_message=commit_msg
            )
        
        done.set()
        tracker.report(100)
        return True
    
    except Exception as e:
        if 'done' in locals():
            done.set()
        log.error(f"Upload failed: {e}")
        return False