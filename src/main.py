#!/usr/bin/env python3

import sys
import time
from pathlib import Path

from src.config import Config
from src.utils import (
    Logger,
    check_for_updates,
    download_model,
    convert_to_f16,
    quantize_model,
    upload_to_hf,
    cleanup_temp
)

log = Logger()

def convert_and_upload(config):
    """Main conversion pipeline"""
    log.section("Starting GGUF Conversion Pipeline")
    
    base_gguf = config.get_output_filename("F16")
    try:
        # Step 1: Download
        if not download_model(config):
            return False
        
        # Step 2: Convert to F16
        if not convert_to_f16(config, base_gguf):
            return False
        
        # Step 3: Upload/Save F16
        upload_to_hf(config, base_gguf, "🟢 Auto-update: GGUF F16")
        
        # Step 4: Quantize and upload/save
        for quant in config.quant_types:
            if quant == "F16":
                continue
            
            if quant not in config.valid_quants:
                log.warning(f"Skipping invalid quant: {quant}")
                continue
            
            output_file = config.get_output_filename(quant)
            
            if quantize_model(config, base_gguf, output_file, quant):
                upload_to_hf(config, output_file, f"🟢 Auto-update: GGUF {quant}")
                
                # Delete after upload (for HF modes)
                if config.upload_mode != "local_only":
                    Path(output_file).unlink()
        
        log.section("Conversion Pipeline Completed!")
        return True
    
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        return False
    
    finally:
        log.info("Cleaning up temporary files...")
        
        # Cleanup temp model download
        cleanup_temp(config.temp_dir)
        
        # Cleanup base F16 (for HF modes only)
        if config.upload_mode != "local_only" and Path(base_gguf).exists():
            Path(base_gguf).unlink()


def main():
    """Main loop"""
    try:
        config = Config()
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)
    
    # Banner
    log.section("GGUF LLMs Auto-Converter v1.2 by Greyscope&Co")
    log.info(f"Repository: {config.repo_id}")
    log.info(f"Upload Mode: {config.upload_mode}")
    
    if config.upload_mode == "new_repo":
        log.info(f"Target Repo: {config.target_repo}")
    elif config.upload_mode == "local_only":
        log.info(f"Output Directory: {config.output_dir}")
        log.info(f"Auto-cleanup: {config.local_cleanup_hours} hours")
    
    log.info(f"Check Interval: {config.check_interval}s ({config.check_interval/60:.1f} min)")
    log.info(f"Output Formats: {', '.join(config.quant_types)}")
    
    if config.base_model_tokenizer:
        log.info(f"Base Tokenizer: {config.base_model_tokenizer}")
    else:
        log.info("Base Tokenizer: Auto-detect or model's own")
    
    print()
    
    last_sha = None
    idle_count = 0
    
    while True:
        try:
            has_update, current_sha = check_for_updates(config, last_sha)
            
            if has_update:
                idle_count = 0
                log.success(f"New commit: {current_sha[:12]}")
                
                if convert_and_upload(config):
                    last_sha = current_sha
                    log.info("⏳ Waiting for next update...")
                else:
                    log.warning("Conversion failed, will retry")
            else:
                idle_count += 1
                if idle_count == 1:
                    log.warning("No updates - IDLE mode")
                elif idle_count % 12 == 0:
                    hrs = (idle_count * config.check_interval) / 3600
                    log.warning(f"IDLE {hrs:.1f}h")
            
            time.sleep(config.check_interval)
        
        except KeyboardInterrupt:
            log.info("\nShutdown signal received")
            log.success("Goodbye!")
            break
        
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.warning("Retry in 60s...")
            time.sleep(60)


if __name__ == "__main__":
    main()