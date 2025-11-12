<h1 align="center">GGUF LLMs Converter for Huggingface Hub Models with Multiple Quantizations (GGUF-Format)</h1>

<p align="center">
  <strong>Automated conversion of any Huggingface model to multiple GGUF LLMs quantization formats</strong><br>
  <em>Supports continuous monitoring, auto-detection, and universal deployment modes</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Release-v1.2-FF0069" alt="Version">
  <img src="https://img.shields.io/badge/GGUF-Available-brightgreen" alt="GGUF">
  <img src="https://img.shields.io/badge/llama.cpp-Compatible-orange" alt="llama.cpp">
  <img src="https://img.shields.io/badge/🤗 Huggingface-Models-blue" alt="Huggingface">
  <a href="https://github.com/arcxteam/gguf-convert-model/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

---

## 📖 Overview

**Universal GGUF LLMs Converter** is a production-ready, Docker-based solution for automatically converting HuggingFace models to GGUF format with multiple quantization types. Built with `llama.cpp` integration and intelligent tokenizer detection, this tool streamlines the conversion workflow for both personal and community models.

### Key Features

- 🔄 **Continuous Monitoring**: Automatically detects and converts new model updates from HuggingFace repositories
- 🤖 **Auto-Detection**: Intelligent tokenizer detection for 50+ popular model architectures (Qwen, Llama, Mistral, Phi, Gemma, etc.)
- 📦 **Multiple Quantization**: Supports F16, F32, BF16, and all K-quant formats (Q2_K to Q8_0)
- 🎯 **Flexible Deploy**: Three (3) upload modes - same repository, new repository, or local-only storage
- 🧹 **Smart Cleanup**: Automatic temporary file management to prevent storage used
- 🐳 **Docker**: Fully container with optimized build times and resource usage
- 📊 **Progress Tracking**: Clean, milestone-based logging with colorized console output

## 🛠️ Requirements

<p>
  <img src="https://img.shields.io/badge/VPS_Server-232F3E?style=for-the-badge&logo=digitalocean&logoColor=red" alt="VPS">
  <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Linux">
  <img src="https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace">
</p>

**System Requirements:**
- Linux-based VPS or local machine
- Docker & Docker Compose installed
- HuggingFace account with **WRITE** access token
- Sufficient disk space for model downloads and conversion (varies by model size)

## 📁 Project Structure

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

## 🚀 **Quick Start**
### 1. Prerequisites

**HuggingFace Access Token:**
- Visit settings → https://huggingface.co/settings/tokens
- Create a new token with **Write** permissions
- Copy the token (starts with `hf_`)

**Install Docker & Compose** <mark>if not already installed</mark>
> Instal docker is optional, if you don't have.. try securely

```
curl -sSL https://raw.githubusercontent.com/arcxteam/succinct-prover/refs/heads/main/docker.sh | sudo bash
```

### 2. Clone Repository

```
git clone https://github.com/arcxteam/gguf-convert-model.git
cd gguf-convert-model
```

### 3. Configure Environment
> Create edit & save configuration file

```
cp .env.example .env
nano .env
```
> Example config environment variable

```diff
# HF token with WRITE permission
HUGGINGFACE_TOKEN=hf_xxxxxxxx

# Source model repository to convert
+ Example: Qwen/Qwen3-0.6B
REPO_ID=username/model-name

# Use interval in secs
+ Default 0 = only one-time convert, for other commits setup more)
CHECK_INTERVAL=0

# Output formats (comma-separated, no spaces)
# Available: F16,BF16,F32,Q2_K,Q2_K_S,Q3_K_S,Q3_K_M,Q3_K_L,Q4_K_S,Q4_K_M,Q4_K_L,Q5_K_S,Q5_K_M,Q5_K_L,Q6_K,Q8_0
+ Recommended: F16,Q4_K_M,Q5_K_M,Q6_K
QUANT_TYPES=F16,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K

# ========================================
# UPLOAD MODE - Choose ONE option below
# ========================================

# OPTION 1: same_repo
# Upload to the same repository as own source model
+ Use this only YOUR OWN models with WRITE access
UPLOAD_MODE=same_repo

# OPTION 2: new_repo
# TARGET_REPO will be auto-generated as: username/ModelName-GGUF
+ Leave TARGET_REPO empty for auto (recommended)
+ Or manually specify: TARGET_REPO=your-username/custom-name-GGUF
UPLOAD_MODE=new_repo
TARGET_REPO=

# OPTION 3: local_only
+ Save to local directory only (no upload hugging)
+ Files auto-delete after LOCAL_CLEANUP_HOURS
UPLOAD_MODE=local_only
OUTPUT_DIR=./output

# Only set if auto-detection fails (default)
+ Example: Qwen/Qwen3-0.6B
BASE_MODEL_TOKENIZER=

# Output filename pattern (default)
# Placeholders: {model_name} = extracted base name, {quant} = format type
+ Result example: Qwen3-0.6B-Instruct-Q4_K_M.gguf
OUTPUT_PATTERN={model_name}-{quant}.gguf

# Auto-cleanup hours (default)
+ Setup you need local_only mode
LOCAL_CLEANUP_HOURS=24

# Timezone
TZ=Asia/Singapore
```

## 📊 **Configuration Reference**

| ENV Variable | Required? | When to Change | Default if Empty |
|--------------|-----------|----------------|------------------|
| `HUGGINGFACE_TOKEN` | ✅ Yes | Always (your token) | `ERROR` |
| `REPO_ID` | ✅ Yes | Always (source model) | `ERROR` |
| `CHECK_INTERVAL` | ⚠️ Optional | Default= 0 or Changes | `in secs 3600=1h` |
| `QUANT_TYPES` | ⚠️ Optional | Change formats needed | `F16,Q4_K_M,Q5_K_M,more` |
| `UPLOAD_MODE` | ⚠️ Optional | Change based on use case | default `new_repo` |
| `TARGET_REPO` | ⚠️ Conditional | Only if `new_repo` mode | Same as `REPO_ID` |
| `OUTPUT_DIR` | ⚠️ Conditional | Only if `local_only` mode | `./output` |
| `BASE_MODEL_TOKENIZER` | ❌ Optional | Only if auto-detect fails | `empty = auto` |
| `OUTPUT_PATTERN` | ❌ Optional | Only if custom naming | `{model_name}-{quant}.gguf` |
| `LOCAL_CLEANUP_HOURS` | ❌ Optional | Only for `local_only` | default `24hour` |
| `TZ` | ❌ Optional | Change to your timezone | UTC |

### ✅ Checklist - What to Change

**Always Change:**
- ✅ `HUGGINGFACE_TOKEN` → Your personal token
- ✅ `REPO_ID` → Model to convert

**Usually Change:**
- ⚠️ `CHECK_INTERVAL` → Frequency (or 0 for one-time)
- ⚠️ `QUANT_TYPES` → Formats you need
- ⚠️ `UPLOAD_MODE` → Based on use case

**Change Only If Needed:**
- ❌ `TARGET_REPO` → If using `new_repo` mode
- ❌ `OUTPUT_DIR` → If using `local_only` mode
- ❌ `BASE_MODEL_TOKENIZER` → If auto-detect fails
- ❌ `OUTPUT_PATTERN` → If custom naming wanted
- ❌ `LOCAL_CLEANUP_HOURS` → If different cleanup time
- ❌ `TZ` → Your timezone (up to you)

**Never Change (Leave Default):**
- ✅ Comments (helpful documentation)
- ✅ Commented-out options (for reference)

### 3.🏃 **Build and Start**
> Starting running

```
docker compose up --build -d
```

> Monitor logs & stop

```
docker compose logs -f
# docker compose down
```

## 📊 **Supported Quantization Formats**

| Format | Precision | Size Reduction | Use Case |
|--------|-----------|----------------|----------|
| **F32** | Full (32-bit) | None | Maximum precision |
| **F16** | Half (16-bit) | ~50% | High quality general use |
| **BF16** | Brain Float 16 | ~50% | Training-optimized |
| **Q8_0** | 8-bit | ~75% | Near-lossless compression |
| **Q6_K** | 6-bit | ~80% | High quality compression |
| **Q5_K_M** | 5-bit | ~83% | **Recommended** balance |
| **Q4_K_M** | 4-bit | ~87% | **Popular** for production |
| **Q3_K_M** | 3-bit | ~90% | Aggressive compression |
| **Q2_K** | 2-bit | ~93% | Maximum compression |

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
