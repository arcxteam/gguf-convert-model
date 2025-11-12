<h1 align="center">GGUF Converter for All Huggingface Hub Models with Multiple Quantizations (GGUF-Format)</h1>

## Support
![VERSION](https://img.shields.io/badge/Release-v1.2-yellow)
![GGUF](https://img.shields.io/badge/GGUF-Available-brightgreen)
![LLAMA](https://img.shields.io/badge/llama.cpp-Available-orange)
![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-blue)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/arcxteam/gguf-convert-model/blob/main/LICENSE)

## Requirements
![VPS](https://img.shields.io/badge/VPS_Server-232F3E?style=for-the-badge&logo=digitalocean&logoColor=red)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Huggingface](https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)


## **1. Structure Folder**

```diff
gguf-convert-model/
├── .env
├── .env.example
├── .gitignore
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── scripts/
│   └── start.sh
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
└── logs/ (auto-created)
```

## **2. Pre-Requirements**

> Get Huggingface API-token mode **WRITE** → `https://huggingface.co/settings/tokens`

### Install Docker & Compose → <mark>if not yet</mark>
> Instal docker is optional, if you don't have.. try docker `shell` securely

```bash
curl -sSL https://raw.githubusercontent.com/arcxteam/succinct-prover/refs/heads/main/docker.sh | sudo bash
```

## **3. Setup Configure**

```bash
git clone https://github.com/arcxteam/gguf-convert-model.git
cd gguf-convert-model
```

> Edit .env file
```
nano .env
```

```bash
# ===== Required Configuration =====
HUGGINGFACE_TOKEN=hf_xxxxxxxx
REPO_ID=your-username/model-name

# ===== Conversion Timer Upload =====
CHECK_INTERVAL=3600

# Quantization types (comma-separated)
# Available: F16,BF16,Q2_K,Q2_K_S,Q3_K_S,Q3_K_M,Q3_K_L,Q4_K_S,Q4_K_M,Q4_K_L,Q5_K_S,Q5_K_M,Q5_K_L,Q6_K,Q8_0
# Recommended: F16,Q4_K_M,Q5_K_M,Q6_K
QUANT_TYPES=F16,Q3_K_M,Q4_K_M,Q5_K_M

# ===== Upload Mode (Pick ONE) =====
# Option 1: same_repo - Upload GGUF files to the same repo as source model
# Best for: Your own models (like Gensyn training)
# Requires: Write access to REPO_ID
UPLOAD_MODE=same_repo

# Option 2: new_repo - Upload to a separate GGUF-specific repo
# Best for: Converting others' models
# Requires: Write access to TARGET_REPO (will auto-create if not exists)
# UPLOAD_MODE=new_repo
# TARGET_REPO=your-username/model-name-GGUF

# Option 3: local_only - Save to local folder, no HuggingFace upload
# Files saved to OUTPUT_DIR with auto-cleanup after 24 hours
# UPLOAD_MODE=local_only
# OUTPUT_DIR=./output

# ===== Optional: Base Model for Tokenizer =====
# If model doesn't have complete tokenizer, specify base model
# Auto-detected for 50+ popular models (Qwen, Llama, Mistral, etc.)
# Example for input: Qwen/Qwen3-0.6B
# Leave empty to use auto-detection or model's own tokenizer
BASE_MODEL_TOKENIZER=

# ===== Optional: Output Filename Pattern =====
# Placeholders outputs: {model_name}-{quant}.gguf → Qwen3-0.6B-Q4_K_M.gguf
OUTPUT_PATTERN={model_name}-{quant}.gguf

# ===== Optional: Local Output Cleanup =====
# Auto-delete local GGUF files after X hours only mode for (local_only)
LOCAL_CLEANUP_HOURS=24

# ===== Timezone =====
# Edit up to you
TZ=Asia/Singapore
```

### 📊 Quick Reference - Use (.ENV)

| ENV Variable | Required? | When to Change | Default if Empty |
|--------------|-----------|----------------|------------------|
| `HUGGINGFACE_TOKEN` | ✅ Yes | Always (your token) | ERROR |
| `REPO_ID` | ✅ Yes | Always (source model) | ERROR |
| `CHECK_INTERVAL` | ⚠️ Optional | Change based on frequency | 3600 (1 hour) |
| `QUANT_TYPES` | ⚠️ Optional | Change formats needed | F16,Q4_K_M,Q5_K_M |
| `UPLOAD_MODE` | ⚠️ Optional | Change based on use case | same_repo |
| `TARGET_REPO` | ⚠️ Conditional | Only if `new_repo` mode | Same as REPO_ID |
| `OUTPUT_DIR` | ⚠️ Conditional | Only if `local_only` mode | ./output |
| `BASE_MODEL_TOKENIZER` | ❌ Optional | Only if auto-detect fails | (empty = auto) |
| `OUTPUT_PATTERN` | ❌ Optional | Only if custom naming | `{model_name}-{quant}.gguf` |
| `LOCAL_CLEANUP_HOURS` | ❌ Optional | Only for `local_only` | 24 |
| `TZ` | ❌ Optional | Change to your timezone | UTC |

### ✅ Checklist - What to Change

### Always Change:
- ✅ `HUGGINGFACE_TOKEN` → Your personal token
- ✅ `REPO_ID` → Model to convert

### Usually Change:
- ⚠️ `CHECK_INTERVAL` → Frequency (or 0 for one-time)
- ⚠️ `QUANT_TYPES` → Formats you need
- ⚠️ `UPLOAD_MODE` → Based on use case (see contoh above)

### Change Only If Needed:
- ❌ `TARGET_REPO` → If using `new_repo` mode
- ❌ `OUTPUT_DIR` → If using `local_only` mode
- ❌ `BASE_MODEL_TOKENIZER` → If auto-detect fails
- ❌ `OUTPUT_PATTERN` → If custom naming wanted
- ❌ `LOCAL_CLEANUP_HOURS` → If different cleanup time
- ❌ `TZ` → Your timezone (cosmetic for logs)

### Never Change (Leave Default):
- ✅ Comments (helpful documentation)
- ✅ Commented-out options (for reference)

## **4. Starting Running**

> Build
```
docker compose up --build -d
```

> Monitor logs & 
```
docker compose logs -f
```
