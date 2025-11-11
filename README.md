<h1 align="center">GGUF Converter for All Huggingface Hub Models with Multiple Quantizations (GGUF-Format)</h1>

## Requirements
![VPS](https://img.shields.io/badge/VPS_Server-232F3E?style=for-the-badge&logo=digitalocean&logoColor=red)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Huggingface](https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

> Get HF token & use mode **WRITE** → `https://huggingface.co/settings/tokens`

### Install Docker & Compose → <mark>if not yet</mark>
> Instal docker is optional, if you not have..try Docker securely

```bash
curl -sSL https://raw.githubusercontent.com/arcxteam/succinct-prover/refs/heads/main/docker.sh | sudo bash
```

### **1. Structure Folder**

```
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

### **2. Setup & Start**

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

**Note; example use search your target and copy as name into env file** 
- https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther

> Build
```
docker compose up --build -d
```

> Monitor logs
```
docker compose logs -f
```
