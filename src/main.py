#!/usr/bin/env python3

import sys
import time
from pathlib import Path
from src.config import Config
from src.utils import (
    Logger,
    check_for_updates,
    download_model,
    convert_to_gguf,
    quantize_model,
    upload_to_hf,
    cleanup_temp
)

log = Logger()

def convert_and_upload(config):
    """Main conversion pipeline with full format support"""
    log.section("Starting GGUF LLMs Conversion Pipeline")
    
    # Non-quantized formats (F16, F32, BF16)
    non_quant_formats = {"F16", "F32", "BF16"}
    # Determine base format for quantization
    base_format = None
    base_file = None
    
    for fmt in config.quant_types:
        if fmt in non_quant_formats:
            base_format = fmt
            break
    
    # default to F16
    if not base_format:
        base_format = "F16"
        log.warning("No base format (F16/F32/BF16) specified, using F16 as base")
    
    base_file = config.get_output_filename(base_format)
    try:
        # Step 1: Download
        if not download_model(config):
            return False
        
        # Step 2: Convert to base format
        if not convert_to_gguf(config, base_file, base_format):
            return False
        
        # Step 3: Upload/Save base format
        upload_to_hf(config, base_file, f"🟢 Auto-update: GGUF {base_format}")
        
        # Step 4: Process all requested formats
        for quant in config.quant_types:
            if quant == base_format:
                continue
            
            # Non-quantized formats (F16, F32, BF16)
            if quant in non_quant_formats:
                output_file = config.get_output_filename(quant)
                if convert_to_gguf(config, output_file, quant):
                    upload_to_hf(config, output_file, f"🟢 Auto-update: GGUF {quant}")
                    if config.upload_mode != "local_only":
                        Path(output_file).unlink()
                continue
            
            # Quantized formats (Q2_K, Q3_K_M, etc.)
            if quant not in config.valid_quants:
                log.warning(f"Skipping invalid quant: {quant}")
                continue
            
            output_file = config.get_output_filename(quant)
            
            if quantize_model(config, base_file, output_file, quant):
                upload_to_hf(config, output_file, f"🟢 Auto-update: GGUF {quant}")
                
                if config.upload_mode != "local_only":
                    Path(output_file).unlink()
        
        log.section("Conversion Pipeline Completed!")
        return True
    
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        return False
    
    finally:
        log.info("Cleaning up temporary files...")
        cleanup_temp(config.temp_dir)
        
        if base_file and config.upload_mode != "local_only" and Path(base_file).exists():
            Path(base_file).unlink()


def main():
    """Main loop bru"""
    try:
        config = Config()
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)
    
    # Banner simple
    log.section("GGUF LLMs Auto-Converter v1.2 by Greyscope&Co")
    log.info(f"Repository: {config.repo_id}")
    log.info(f"Upload Mode: {config.upload_mode}")
    
    if config.upload_mode == "new_repo":
        log.info(f"Target Repo: {config.target_repo}")
    elif config.upload_mode == "local_only":
        log.info(f"Output Directory: {config.output_dir}")
        log.info(f"Auto-cleanup: {config.local_cleanup_hours}h")
    
    if config.check_interval > 0:
        log.info(f"Check Interval: {config.check_interval}s ({config.check_interval/60:.1f} min)")
    else:
        log.info("Check Interval: Disabled (one-time conversion)")
    
    log.info(f"Output Formats: {', '.join(config.quant_types)}")
    
    if config.base_model_tokenizer:
        log.info(f"Base Tokenizer: {config.base_model_tokenizer}")
    else:
        log.info("Base Tokenizer: Auto-detect")
    
    print()
    
    # One-time CHECK_INTERVAL=0
    if config.check_interval <= 0:
        log.warning("One-time conversion mode (CHECK_INTERVAL=0)")
        convert_and_upload(config)
        log.success("Conversion complete. Exiting.")
        sys.exit(0)
    
    # Continuous monitoring mode
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
                    log.info("🕗 Waiting for next update...")
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
            log.info("\n🔴 Shutdown signal received")
            log.success("🔵 Say Goodbye!")
            break
        
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.warning("Retry in 60s...")
            time.sleep(60)


if __name__ == "__main__":
    main()
