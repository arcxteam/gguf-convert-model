#!/bin/bash
set -e

echo "=========================================="
echo " ✳️ GGUF LLMs Auto-Converter v1.2 ❇️"
echo "=========================================="
echo "Repository: $REPO_ID"
echo "Upload Mode: $UPLOAD_MODE"
echo "Check Interval: ${CHECK_INTERVAL}s"
echo "Quant Types: $QUANT_TYPES"
echo "=========================================="
echo ""

# Validate required env vars
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "❌ ERROR: HUGGINGFACE_TOKEN is not set!"
    exit 1
fi

if [ -z "$REPO_ID" ]; then
    echo "❌ ERROR: REPO_ID is not set!"
    exit 1
fi

# Run main script
exec python3 -m src.main
