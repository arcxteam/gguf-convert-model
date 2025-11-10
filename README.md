## Requirements
![VPS](https://img.shields.io/badge/VPS-CPU/GPU-232F3E?style=for-the-badge&logo=digitalocean&logoColor=green)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Huggingface](https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

> Get your HF token https://huggingface.co/settings/tokens

## Install Docker & Compose → <mark>if not yet</mark>
> Instal docker is optional, if you not have..try Docker securely

```bash
curl -sSL https://raw.githubusercontent.com/arcxteam/succinct-prover/refs/heads/main/docker.sh | sudo bash
```

## Docker Setup 

### **1. Structure Folder**

```
gguf-convert-model/
├── .env
├── docker-compose.yml
├── Dockerfile
├── convert_gguf.py
├── requirements.txt
└── start.sh
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
# HuggingFace Configuration
HUGGINGFACE_ACCESS_TOKEN=hf_xxxxxxxxx
REPO_ID=0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther

# Conversion Settings
CHECK_INTERVAL=3600
QUANT_TYPES=F16,Q3_K_M,Q4_K_M,Q5_K_M

# Timezone
TZ=Asia/Jakarta
```

**Note; example use search your target and copy as name into env file** 
> `https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther`

> Build
```
docker compose up --build -d
```

> Monitor logs
```
docker compose logs -f
```
