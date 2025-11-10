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

> Build
```
docker compose up --build -d
```

> Monitor logs
```
docker compose logs -f
```
