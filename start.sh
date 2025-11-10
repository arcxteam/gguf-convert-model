#!/bin/bash
set -e

echo "=================================================="
echo "🚀 GGUF Auto-Converter for All Huggingface"
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
exec python3 /app/convert_gguf.py
