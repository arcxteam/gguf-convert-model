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
  results:
  - task:
      type: text-generation
      name: Mathematical Reasoning
    dataset:
      name: Composite Reasoning Dataset
      type: custom
    metrics:
    - type: training_rounds
      value: 43610
      name: Completed Training Rounds
    - type: total_rounds
      value: 100000
      name: Target Rounds
    - type: progress
      value: 43.61
      name: Training Progress (%)
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

This model is a continuously trained Qwen3-0.6B fine-tuned using **Gensyn RL-Swarm** framework with **GRPO (Generalized Reward Policy Optimization)** for enhanced reasoning and mathematical capabilities. **Note: Current training focuses on math/reasoning tasks**.

**Agent ID:** `tall_tame_panther`  
**Training Status:** 🟢 LIVE - Model updates automatically every 5-10 minutes  
**Current Progress:** Round 43,610+ / 100,000 (43,61%)  
**Framework Version:** Gensyn RL-Swarm v0.6.4 
**Contract:** SwarmCoordinator v0.4.2

## Key Features

- **Real-time Training**: Continuous learning with distributed RL across Gensyn swarm network
- **Multi-domain Reasoning**: Trained on logic, arithmetic, and mathematical problem-solving
- **GGUF Support**: Multiple quantized formats available (F16, Q3_K_M, Q4_K_M, Q5_K_M)
- **llama.cpp Compatible**: Ready for edge deployment and local inference
- **BF16 Precision**: Trained with bfloat16 for optimal performance
- **TGI Compatible**: Supports Text Generation Inference for production deployment
- **Chat Format Support**: Inherits Qwen3 chat template for conversational use

## Training Data

The model is trained on a composite dataset (1,000 samples) with weighted sampling strategy:

| Dataset | Weight | Focus Area |
|---------|--------|------------|
| Propositional Logic | 7 | Logical reasoning, truth tables, Boolean operations |
| Calendar Arithmetic | 6 | Date calculations, leap years, recurring events |
| Decimal Arithmetic | 5 | Multi-term decimal operations with precision |
| Base Conversion | 4 | Number system conversions (base 2-16) |
| Fraction Simplification | 4 | GCD/LCM, fraction reduction |
| Basic Arithmetic | 2 | Foundation operations with parentheses |

**Total Dataset Size:** 1,000 composite samples  
**Training Samples per Round:** 2  
**Evaluation:** Real-time via swarm coordination

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

prompt = "What is 3/4 simplified to lowest terms?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=256, temperature=0.6, top_p=0.95)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```

### Chat Format (Conversational)

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther")
tokenizer = AutoTokenizer.from_pretrained("0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther")

messages = [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "Explain how to simplify 24/36 step by step."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs))
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

### GGUF Quantization Strategy

The Q5_K_M format uses mixed precision for optimal quality:

- **Token Embeddings**: Q6_K (high quality vocab representation)
- **Attention Weights**: Q5_K (balanced quality/size)
- **Feed-Forward**: Q5_K/Q6_K (mixed for optimal performance)
- **Layer Norms**: F32 (full precision for stability)

This strategy ensures minimal quality loss while maintaining small file size.

## Chat Format & Conversational Use

This model inherits **Qwen3's chat template** for structured conversations.

### Format Structure

```
<|im_start|>system
{system_message}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
{assistant_response}
<|im_end|>
```

### Chat Template Features

- **System Instructions**: Guide model behavior with system messages
- **Multi-turn Dialogue**: Maintains conversation context
- **Tool Calling**: Support function calling (if enabled in training)
- **Reasoning Mode**: `<think>` tags for chain-of-thought (experimental)

**Note**: While the model supports chat format structurally, optimal conversational performance depends on whether training data included formatted dialogues. Current training focuses on **math/reasoning tasks**.

## Training Configuration

### Gensyn RL-Swarm Architecture

```
Training Framework:
  Method: GRPO (Generalized Reward Policy Optimization)
  Base Model: Qwen/Qwen3-0.6B
  Training Regime: bfloat16 mixed precision
  Max Rounds: 100,000
  Update Frequency: Every 5-10 minutes
  Generations per Round: 2
  Seed: 42

Blockchain Integration:
  Network: Gensyn Testnet
  Chain ID: 685685
  Contract: SwarmCoordinator v0.4.2

Swarm Communication:
  Framework: Hivemind P2P Backend
  Initial Peers: 3 bootnodes
  Beam Size: 30

Reward System:
  Manager: DefaultRewardManager
  Reward Function: RGRewards (Reasoning Gym)
  Judge API: https://swarm-judge.internal-apps-central1.clusters.gensyn.ai
```

### Model Hyperparameters

```
Architecture:
  Hidden Size: 1024
  Intermediate Size: 3072
  Layers: 28
  Attention Heads: 16
  KV Heads: 8
  Head Dimension: 128
  Context Length: 40,960 tokens
  Vocabulary: 151,936 tokens

GRPO Config:
  Epsilon: 0.2
  Epsilon High: 0.28
  Gradient Checkpointing: Enabled
  
Generation:
  Temperature: 0.6
  Top-K: 20
  Top-P: 0.95
```

## Model Capabilities

This model excels at:

1. **Logical Reasoning**: Propositional logic, truth evaluation, Boolean algebra
2. **Mathematical Operations**: Multi-precision arithmetic, decimal calculations, fractions
3. **Number Systems**: Base conversion (binary, octal, decimal, hexadecimal)
4. **Date/Time Calculations**: Calendar arithmetic, leap years, day-of-week
5. **Step-by-step Problem Solving**: Chain-of-thought reasoning
6. **Conversational Tutoring**: Interactive problem-solving (via chat format)

## Limitations

- **Specialized Domain**: Optimized for reasoning/math; may underperform on creative writing
- **Training in Progress**: Weights update every 5-10 minutes; performance varies
- **Scale**: 0.6B parameters - suitable for edge but not SOTA for complex reasoning
- **Experimental**: Decentralized RL training; behavior less predictable than supervised models
- **Context**: Best performance within 4K tokens (full 40K supported)

## Update Schedule

| Format | Frequency | Trigger |
|--------|-----------|---------|
| Safetensors (BF16) | Every 5-10 min | Automatic via RL-Swarm |
| GGUF (all formats) | Every 1 hour | Auto-conversion pipeline |

**Auto-Conversion Pipeline:**
1. Monitors repo for new training commits
2. Downloads latest `model.safetensors`
3. Converts to F16 GGUF base
4. Quantizes to Q3_K_M, Q4_K_M, Q5_K_M
5. Uploads all formats

Check commit history for exact timestamps.

## Gensyn RL-Swarm Technical Details

### Architecture Components

1. **Game Manager**: Orchestrates training rounds and swarm coordination
2. **Trainer**: GRPO implementation for policy optimization
3. **Data Manager**: Dataset loading and weighted sampling
4. **Reward Manager**: Computes rewards via judge API
5. **Coordinator**: Blockchain integration for swarm state
6. **P2P Backend**: Hivemind DHT for model sharing

### Training Process

```
1. Agent joins swarm via P2P network
2. Coordinator assigns round via smart contract
3. Agent samples data from weighted datasets
4. Model generates 2 responses
5. Judge API evaluates and assigns rewards
6. GRPO updates policy based on rewards
7. Updated model shared via DHT
8. Best checkpoint saved to HuggingFace
9. Repeat
```

### Decentralization Benefits

- **Fault Tolerance**: Multiple agents; no single point of failure
- **Diverse Exploration**: Different agents explore different strategies
- **Collective Intelligence**: Agents learn from each other
- **Transparent**: All rounds verified on-chain

**Swarm Agent:** `tall_tame_panther`  
**Contract:** SwarmCoordinator v0.4.2

## Technical Specifications

### Software Stack

- **Framework**: Gensyn RL-Swarm v0.6.4
- **Library**: transformers v4.51+
- **P2P**: hivemind
- **Blockchain**: Gensyn testnet
- **Config**: Hydra + OmegaConf
- **Logging**: WandB integration

### Hardware Requirements

**Training GPU:**
- GPU: NVIDIA 4090 24GB+ (BF16 training)
- RAM: 16GB+
- Cores: 10+
- Storage: 50GB SSD
- Network: High bandwidth for P2P

**Training CPU Optimize:**
- CPU: INTEL or AMD
- Cores: 10+
- RAM: 16GB+
- Storage: 50GB SSD
- Network: High bandwidth for P2P
 
**Inference:**
- Safetensors: 8GB VRAM (GPU) / 16GB RAM (CPU)
- GGUF Q4_K_M: 2GB VRAM (GPU) / 4GB RAM (CPU)
- GGUF Q3_K_M: 3GB RAM (CPU-only)

## Evaluation

### Training Progress Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Completed Rounds | 43,610+ | 100,000 |
| Training Progress | 43.61% | 100% |
| Update Frequency | 5-10 min | Continuous |

**Note**: Formal evaluation benchmarks (GSM8K, MATH, etc.) will be added as training progresses. Current metrics track training rounds completed in the decentralized swarm.

## Reproducibility

To reproduce training:

1. Clone Gensyn RL-Swarm repository
2. Install: `pip install -r requirements.txt`
3. Configure `rgym_exp/config/rg-swarm.yaml`
4. Configure `rgym_exp/src/datasets.yaml`
5. Set environment variables:
```
export HUGGINGFACE_ACCESS_TOKEN=<token>
export MODEL_NAME=Qwen/Qwen3-0.6B
export ORG_ID=<org-id>
export SWARM_CONTRACT=<contract-address>
```
6. Run: `bash run_rl_swarm.sh`

**Note**: Exact reproduction requires same seed (42), dataset config, and swarm state.

## Citation

```
@misc{qwen3-gensyn-swarm-2025,
  author = {0xgrey},
  title = {Qwen3-0.6B-Gensyn-Swarm: Continuous RL Training on Distributed Swarm},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther}},
  note = {Agent ID: tall\_tame\_panther}
}

@misc{gensyn-rl-swarm-2025,
  title = {Gensyn RL-Swarm: Decentralized Reinforcement Learning Framework},
  author = {Gensyn AI},
  year = {2025},
  url = {https://gensyn.ai}
}
```

## References

- **Gensyn Documentation**: https://docs.gensyn.ai/
- **Gensyn GitHub**: https://github.com/gensyn-ai
- **RL-Swarm Contracts**: https://github.com/gensyn-ai/rl-swarm-contracts
- **Qwen3 Model Card**: https://huggingface.co/Qwen/Qwen3-0.6B
- **arXiv:1910.09700**: ML Carbon Emissions methodology

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Contact

- **Developer**: 0xgrey
- **Agent ID**: tall_tame_panther
- **Community**: [Gensyn Discord](https://discord.gg/gensyn)

---

**⚠️ Important**: This is a continuously trained model. For reproducibility, specify commit hash:

```
git clone https://huggingface.co/0xgr3y/Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther
cd Qwen3-0.6B-Gensyn-Swarm-tall_tame_panther
git checkout <commit-hash>
```

---

<div align="center">

**🤖 Trained with ❤️ using Gensyn RL-Swarm**

[![Gensyn](https://img.shields.io/badge/Powered%20by-Gensyn%20AI-orange?style=for-the-badge)](https://gensyn.ai)

</div>