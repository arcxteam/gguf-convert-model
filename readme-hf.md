---
library_name: transformers
tags:
- text-generation
- qwen3
- rl-swarm
- genrl-swarm
- grpo
- gensyn
- trl
- reasoning
- math
- logic
- continuous-training
- reinforcement-learning
- safetensors
- gguf
- conversational
- text-generation-inference
pipeline_tag: text-generation
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3-0.6B
datasets:
- propositional_logic
- calendar_arithmetic
- decimal_arithmetic
- base_conversion
- fraction_simplification
- basic_arithmetic
inference: true
model-index:
- name: Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther
  results: []
widget:
- text: "What is 15 * 23?"
  example_title: "Basic Arithmetic"
- text: "Convert decimal 255 to hexadecimal."
  example_title: "Base Conversion"
- text: "Simplify the fraction 24/36."
  example_title: "Fraction Simplification"
---

# Qwen3-0.6B-Gensyn-Swarm (tall_tame_panther)

[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther)
[![GGUF](https://img.shields.io/badge/GGUF-Available-green)](https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther/tree/main)
[![Gensyn](https://img.shields.io/badge/Trained%20with-Gensyn%20RL--Swarm-orange)](https://gensyn.ai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Model Overview

This model is a continuously trained Qwen3-0.6B fine-tuned using **Gensyn RL-Swarm** framework with **GRPO (Generalized Reward Policy Optimization)** for enhanced reasoning and mathematical capabilities.

**Agent ID:** `tall_tame_panther`  
**Training Status:** 🔴 LIVE - Model updates automatically every 5-10 minutes  
**Current Progress:** Round 43610+ / 1,000,000  
**Framework Version:** Gensyn RL-Swarm v0.4.2  
**Contract:** SwarmCoordinator v0.4.2

## Key Features

- **Real-time Training**: Continuous learning with distributed RL across Gensyn swarm network
- **Multi-domain Reasoning**: Trained on logic, arithmetic, and mathematical problem-solving
- **GGUF Support**: Multiple quantized formats available (F16, Q3_K_M, Q4_K_M, Q5_K_M)
- **llama.cpp Compatible**: Ready for edge deployment and local inference
- **BF16 Precision**: Trained with bfloat16 for optimal performance
- **TGI Compatible**: Supports Text Generation Inference for production deployment
- **Conversational**: Can be used for interactive reasoning tasks

## Training Data

The model is trained on a composite dataset (1,000 samples) with weighted sampling strategy defined in `datasets.yaml`:

| Dataset | Weight | Samples | Focus Area |
|---------|--------|---------|------------|
| Propositional Logic | 7 | 500 | Logical reasoning, truth tables, Boolean operations |
| Calendar Arithmetic | 6 | 500 | Date calculations, leap years, recurring events |
| Decimal Arithmetic | 5 | 500 | Multi-term decimal operations with precision |
| Base Conversion | 4 | 500 | Number system conversions (base 2-16) |
| Fraction Simplification | 4 | 500 | GCD/LCM, fraction reduction |
| Basic Arithmetic | 2 | 500 | Foundation operations with parentheses |

**Total Dataset Size:** 1,000 composite samples  
**Training Samples per Round:** 2  
**Evaluation Samples:** Real-time via swarm coordination

### Dataset Configuration Details

```
# From rgym_exp/src/datasets.yaml
Propositional Logic:
  - Variables: 2-4
  - Statements: 2-4
  - Complexity: 1-3

Calendar Arithmetic:
  - Year: 2023
  - Offset: up to 100 days
  - Leap year range: 200 years
  - Tasks: count_days, weekday_of_date, is_leap_year, recurring_event_day

Decimal Arithmetic:
  - Terms: 2-6
  - Decimal places: 1-3
  - Precision: 5

Base Conversion:
  - Base range: 2-16
  - Value range: 0-1000

Fraction Simplification:
  - Value range: 1-100
  - Factor range: 2-100
  - Styles: plain, latex_frac, latex_dfrac

Basic Arithmetic:
  - Terms: 2-6
  - Digits: 1-4
  - Operators: +, -, *, /
  - Parentheses: enabled
```

## Quick Start

### Standard Transformers

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther")

# Example: Math reasoning
prompt = "What is 3/4 simplified to lowest terms?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=256, temperature=0.6, top_p=0.95)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```

### Text Generation Inference (TGI)

```
docker run -d --gpus all \
  -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id 0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther \
  --max-input-length 4096 \
  --max-total-tokens 8192
```

### GGUF with llama.cpp

```
# Download quantized model (recommended: Q4_K_M)
wget https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther/resolve/main/Qwen3-0.6B-Gensyn-Swarm-Q4_K_M.gguf

# Run inference
./llama-cli -m Qwen3-0.6B-Gensyn-Swarm-Q4_K_M.gguf \
  -p "Solve: (5 + 3) * 2 = ?" \
  --temp 0.6 --top-p 0.95
```

### Ollama

```
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./Qwen3-0.6B-Gensyn-Swarm-Q4_K_M.gguf
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
SYSTEM "You are a helpful assistant specialized in mathematical reasoning and logic."
EOF

# Create and run
ollama create qwen3-swarm -f Modelfile
ollama run qwen3-swarm "What is 15 multiplied by 23?"
```

## Available Formats

| Format | Size | Precision | Use Case | Download |
|--------|------|-----------|----------|----------|
| Safetensors (BF16) | 1.19 GB | BF16 | Full precision training/fine-tuning | `model.safetensors` |
| GGUF F16 | 1.14 GB | FP16 | High quality inference | `Qwen3-0.6B-Gensyn-Swarm-F16.gguf` |
| GGUF Q5_K_M | 444 MB | 5-bit | Balanced quality/size | `Qwen3-0.6B-Gensyn-Swarm-Q5_K_M.gguf` |
| GGUF Q4_K_M | 397 MB | 4-bit | **Recommended** for production | `Qwen3-0.6B-Gensyn-Swarm-Q4_K_M.gguf` |
| GGUF Q3_K_M | 347 MB | 3-bit | Smallest, fastest | `Qwen3-0.6B-Gensyn-Swarm-Q3_K_M.gguf` |

All GGUF formats are **llama.cpp compatible** and auto-updated hourly.

## Training Configuration

### Gensyn RL-Swarm Architecture

The model is trained using a decentralized reinforcement learning framework with the following components:

```
# From rgym_exp/config/rg-swarm.yaml

Training Framework:
  Method: GRPO (Generalized Reward Policy Optimization)
  Base Model: Qwen/Qwen3-0.6B
  Training Regime: bfloat16 mixed precision
  Max Rounds: 1,000,000
  Max Stage: 1
  Update Frequency: Every 5-10 minutes
  Generations per Round: 2
  Transplant Trees: 1
  Seed: 42

Blockchain Integration:
  Network: Gensyn Testnet
  Chain ID: 685685
  RPC: https://gensyn-testnet.g.alchemy.com/public
  Contract: SwarmCoordinator v0.4.2
  Modal Proxy: http://localhost:3000/api/

Swarm Communication:
  Framework: Hivemind P2P Backend
  Initial Peers: 3 bootnodes
  Bootnodes:
    - /ip4/38.101.215.12/tcp/30011/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ
    - /ip4/38.101.215.13/tcp/30012/p2p/QmWhiaLrx3HRZfgXc2i7KW5nMUNK7P9tRc71yFJdGEZKkC
    - /ip4/38.101.215.14/tcp/30013/p2p/QmQa1SCfYTxx7RvU7qJJRo79Zm1RAwPpkeLueDVJuBBmFp
  Startup Timeout: 120s
  Beam Size: 25

Reward System:
  Manager: DefaultRewardManager
  Function Store: RoundRewardFnStore
  Reward Function: RGRewards (Reasoning Gym Rewards)
  Judge: Swarm Judge API (https://swarm-judge.internal-apps-central1.clusters.gensyn.ai)
```

### Training Hyperparameters

```
Model Architecture:
  Hidden Size: 1024
  Intermediate Size: 3072
  Num Hidden Layers: 28
  Num Attention Heads: 16
  Num Key-Value Heads: 8
  Head Dimension: 128
  Max Position Embeddings: 40,960
  RMS Norm Epsilon: 1e-06
  Rope Theta: 1,000,000
  Vocabulary Size: 151,936

GRPO Trainer Config:
  Epsilon: 0.2
  Epsilon High: 0.28
  Generations: 2
  Gradient Checkpointing: Enabled
  Learning Rate: Adaptive
  
Generation Config:
  Temperature: 0.6
  Top-K: 20
  Top-P: 0.95
  BOS Token: 151643
  EOS Token: 151645
  Pad Token: 151643
```

## Model Capabilities

This model excels at:

1. **Logical Reasoning**: Propositional logic, truth evaluation, Boolean algebra, logical equivalences
2. **Mathematical Operations**: Multi-precision arithmetic, decimal calculations, fraction manipulation
3. **Number Systems**: Base conversion between binary, octal, decimal, hexadecimal
4. **Date/Time Calculations**: Calendar arithmetic, leap year detection, day-of-week calculations
5. **Step-by-step Problem Solving**: Chain-of-thought reasoning for complex multi-step tasks
6. **Conversational Math Tutoring**: Interactive problem-solving guidance

## Limitations

- **Specialized Domain**: Optimized for reasoning/math tasks; may underperform on creative writing or general chat
- **Training in Progress**: Model weights update every 5-10 minutes; performance may vary between checkpoints
- **Scale**: 0.6B parameters - suitable for edge devices but not state-of-the-art for complex reasoning
- **Experimental**: Trained via decentralized RL swarm; behavior may be less predictable than supervised models
- **Context Length**: 40K tokens supported but best performance within 4K tokens

## Update Schedule

| Format | Update Frequency | Trigger |
|--------|------------------|---------|
| Safetensors (BF16) | Every 5-10 minutes | Automatic via RL-Swarm training |
| GGUF variants (all) | Every 1 hour | Automatic conversion from latest checkpoint |

**Auto-Conversion Pipeline:**
- Monitors repo for new training commits
- Downloads latest `model.safetensors`
- Converts to F16 GGUF base
- Quantizes to Q3_K_M, Q4_K_M, Q5_K_M
- Uploads all formats to repo

Check commit history for exact timestamps of each update.

## Gensyn RL-Swarm Technical Details

This model is trained using [Gensyn RL-Swarm](https://gensyn.ai), a decentralized reinforcement learning framework:

### Architecture Components

1. **Game Manager** (`rgym_exp/src/manager.py`): Orchestrates training rounds and swarm coordination
2. **Trainer** (`rgym_exp/src/trainer.py`): GRPO implementation for policy optimization
3. **Data Manager** (`rgym_exp/src/data.py`): Handles dataset loading and sampling
4. **Reward Manager** (`rgym_exp/src/rewards.py`): Computes rewards using judge API
5. **Coordinator** (`rgym_exp/src/coordinator.py`): Blockchain integration for swarm state
6. **Communication Backend**: Hivemind DHT for peer-to-peer model sharing

### Training Process

```
1. Agent joins swarm via P2P network
2. Coordinator assigns training round via smart contract
3. Agent samples data from weighted datasets
4. Model generates responses (2 generations)
5. Judge API evaluates quality and assigns rewards
6. GRPO updates policy based on rewards
7. Updated model shared via DHT to swarm
8. Best model checkpoint saved to HuggingFace
9. Repeat for next round
```

### Decentralization Benefits

- **Fault Tolerance**: Multiple agents contribute; single node failure doesn't stop training
- **Diverse Exploration**: Different agents explore different strategies
- **Collective Intelligence**: Agents learn from each other's experiences
- **Transparent Verification**: All training rounds verified on-chain

**Swarm Agent:** `tall_tame_panther`  
**Contract:** SwarmCoordinator v0.4.2  
**Testnet Explorer:** https://gensyn-testnet.explorer.com

## Technical Specifications

### Software Stack

- **Training Framework**: Gensyn RL-Swarm v0.4.2
- **Base Library**: transformers v4.51.3
- **Communication**: hivemind (P2P backend)
- **Blockchain**: Web3.py (Gensyn testnet)
- **Configuration**: Hydra + OmegaConf
- **Logging**: WandB integration

### Hardware Requirements

**Training Node:**
- GPU: NVIDIA A100 40GB or equivalent (for BF16 training)
- RAM: 32GB+ system memory
- Storage: 50GB SSD
- Network: High bandwidth for P2P swarm communication

**Inference:**
- Safetensors: 8GB+ VRAM (GPU), 16GB+ RAM (CPU)
- GGUF Q4_K_M: 4GB RAM (CPU), 2GB VRAM (GPU)
- GGUF Q3_K_M: 3GB RAM (CPU-only compatible)

## Reproducibility

To reproduce training results:

1. Clone Gensyn RL-Swarm repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `rgym_exp/config/rg-swarm.yaml` with your settings
4. Set environment variables:
   ```
   export HUGGINGFACE_ACCESS_TOKEN=<your-token>
   export MODEL_NAME=Qwen/Qwen3-0.6B
   export ORG_ID=<your-org-id>
   export SWARM_CONTRACT=<contract-address>
   ```
5. Run: `bash run_rl_swarm.sh`

**Note:** Exact reproduction requires same seed (42), dataset configuration, and swarm coordination state.

## Citation

```
@misc{qwen3-gensyn-swarm-2025,
  author = {0xgr3y},
  title = {Qwen3-0.6B-Gensyn-Swarm: Continuous RL Training on Distributed Swarm},
  year = {2025},
  publisher = {HuggingFace},
  journal = {HuggingFace Model Hub},
  howpublished = {\url{https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther}},
  note = {Agent ID: tall\_tame\_panther}
}

@misc{gensyn-rl-swarm-2025,
  title = {Gensyn RL-Swarm: Decentralized Reinforcement Learning Framework},
  author = {Gensyn AI},
  year = {2025},
  url = {https://gensyn.ai},
  note = {SwarmCoordinator v0.4.2}
}

@article{lacoste2019quantifying,
  title={Quantifying the Carbon Emissions of Machine Learning},
  author={Lacoste, Alexandre and others},
  journal={arXiv preprint arXiv:1910.09700},
  year={2019}
}
```

## References

- **arXiv:1910.09700** - ML Carbon Emissions methodology
- **Gensyn Documentation**: https://docs.gensyn.ai
- **Qwen3 Model Card**: https://huggingface.co/Qwen/Qwen3-0.6B
- **Technical Report**: See `technical_report.pdf` in training repository

## License

Apache 2.0 - See [LICENSE](LICENSE) for details

## Contact & Support

- **Developer**: 0xgr3y
- **Agent ID**: tall_tame_panther
- **Issues**: Open an issue on this repo
- **Community**: [Gensyn Discord](https://discord.gg/gensyn)

---

**⚠️ Important Note**: This is a continuously trained model. For reproducibility, always specify the exact commit hash:

```
# Download specific checkpoint
git clone https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther
cd Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther
git checkout <commit-hash>
```

---

<div align="center">

**🤖 Trained with ❤️ using Gensyn RL-Swarm**

[![Gensyn](https://img.shields.io/badge/Powered%20by-Gensyn%20AI-orange?style=for-the-badge)](https://gensyn.ai)

</div>
```