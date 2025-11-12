import os
import json
import re
from pathlib import Path
from typing import Optional

class Config:
    """Configuration with automatic base model detection"""
    
    # Extended base model mapping (50+ models)
    BASE_MODEL_MAPPING = {
        # Qwen family
        "Qwen2ForCausalLM": "Qwen/Qwen2-0.5B",
        "Qwen3ForCausalLM": "Qwen/Qwen3-0.6B",
        "QWenLMHeadModel": "Qwen/Qwen-7B",
        
        # Llama family
        "LlamaForCausalLM": "meta-llama/Llama-3.1-8B",
        "Llama2ForCausalLM": "meta-llama/Llama-2-7b-hf",
        "Llama3ForCausalLM": "meta-llama/Llama-3.1-8B",
        
        # Mistral family
        "MistralForCausalLM": "mistralai/Mistral-7B-v0.1",
        "MixtralForCausalLM": "mistralai/Mixtral-8x7B-v0.1",
        
        # Phi family (Microsoft)
        "PhiForCausalLM": "microsoft/phi-2",
        "Phi3ForCausalLM": "microsoft/Phi-3-mini-4k-instruct",
        "Phi3SmallForCausalLM": "microsoft/Phi-3-small-8k-instruct",
        "Phi3MediumForCausalLM": "microsoft/Phi-3-medium-4k-instruct",
        
        # Gemma family (Google)
        "GemmaForCausalLM": "google/gemma-2b",
        "Gemma2ForCausalLM": "google/gemma-2-2b",
        "Gemma27BForCausalLM": "google/gemma-2-7b",
        
        # GPT family
        "GPT2LMHeadModel": "gpt2",
        "GPTNeoForCausalLM": "EleutherAI/gpt-neo-2.7B",
        "GPTNeoXForCausalLM": "EleutherAI/gpt-neox-20b",
        "GPTJForCausalLM": "EleutherAI/gpt-j-6b",
        
        # Falcon
        "FalconForCausalLM": "tiiuae/falcon-7b",
        "RWForCausalLM": "tiiuae/falcon-7b",
        
        # Bloom
        "BloomForCausalLM": "bigscience/bloom-7b1",
        
        # MPT (MosaicML)
        "MPTForCausalLM": "mosaicml/mpt-7b",
        
        # StableLM
        "StableLMEpochForCausalLM": "stabilityai/stablelm-3b-4e1t",
        "StableLmForCausalLM": "stabilityai/stablelm-base-alpha-7b",
        
        # Yi (01.AI)
        "YiForCausalLM": "01-ai/Yi-6B",
        
        # Baichuan
        "BaichuanForCausalLM": "baichuan-inc/Baichuan-7B",
        "Baichuan2ForCausalLM": "baichuan-inc/Baichuan2-7B-Base",
        
        # InternLM
        "InternLMForCausalLM": "internlm/internlm-7b",
        "InternLM2ForCausalLM": "internlm/internlm2-7b",
        
        # ChatGLM
        "ChatGLMModel": "THUDM/chatglm-6b",
        "ChatGLM2Model": "THUDM/chatglm2-6b",
        "ChatGLM3Model": "THUDM/chatglm3-6b",
        
        # Others
        "CohereForCausalLM": "CohereForAI/c4ai-command-r-v01",
        "DbrxForCausalLM": "databricks/dbrx-instruct",
        "OlmoForCausalLM": "allenai/OLMo-7B",
        "PersimmonForCausalLM": "adept/persimmon-8b-base",
        "SmolLMForCausalLM": "HuggingFaceTB/SmolLM-135M",
        "GPTBigCodeForCausalLM": "bigcode/starcoder",
        "DeepseekForCausalLM": "deepseek-ai/deepseek-llm-7b-base",
    }
    
    # Valid quantization types
    VALID_QUANTS = {
        "F16", "BF16", "F32",
        "Q2_K", "Q2_K_S",
        "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_L",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q5_K_L",
        "Q6_K", "Q8_0"
    }
    
    def __init__(self):
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.repo_id = os.getenv("REPO_ID")
        
        if not self.hf_token or not self.repo_id:
            raise ValueError("HUGGINGFACE_TOKEN and REPO_ID are required")
        
        self.check_interval = int(os.getenv("CHECK_INTERVAL", "3600"))
        quant_str = os.getenv("QUANT_TYPES", "F16,Q4_K_M,Q5_K_M")
        self.quant_types = [q.strip() for q in quant_str.split(",") if q.strip()]
        
        invalid = [q for q in self.quant_types if q not in self.VALID_QUANTS]
        if invalid:
            raise ValueError(f"Invalid quant types: {invalid}")
        
        self.upload_mode = os.getenv("UPLOAD_MODE", "same_repo").lower()
        if self.upload_mode not in ["same_repo", "new_repo", "local_only"]:
            raise ValueError(f"Invalid UPLOAD_MODE: {self.upload_mode}")
        
        self.target_repo = self._generate_target_repo()
        self.base_model_tokenizer = self._detect_base_model()
        
        self.output_pattern = os.getenv("OUTPUT_PATTERN", "{model_name}-{quant}.gguf")
        self.conversion_timeout = int(os.getenv("CONVERSION_TIMEOUT", "3600"))
        self.local_cleanup_hours = int(os.getenv("LOCAL_CLEANUP_HOURS", "24"))
        
        self.llama_cpp_path = Path(os.getenv("LLAMA_CPP_PATH", "/app/llama.cpp"))
        self.temp_dir = Path(os.getenv("TEMP_DIR", "/tmp/gguf_conversion"))
        self.cache_dir = Path(os.getenv("CACHE_DIR", "/app/cache"))
        self.log_dir = Path(os.getenv("LOG_DIR", "logs"))
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.upload_mode == "local_only":
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    def _extract_base_model_name(self, full_name: str) -> str:
        """
        Universal extraction - works for ANY model name pattern.
        
        Pattern: Extract ModelName-Version-Variant (first 3 components)
        
        Examples:
        - Qwen2.5-0.5B-Instruct-Gensyn-Swarm-xxx → Qwen2.5-0.5B-Instruct
        - Llama-3.1-8B-finetune-custom → Llama-3.1-8B
        - Mistral-7B-v0.1-trained → Mistral-7B-v0.1
        """
        
        # Remove common training/finetune suffixes (universal patterns)
        patterns_to_remove = [
            r'-finetune.*',
            r'-finetuned.*',
            r'-trained.*',
            r'-custom.*',
            r'-chat$',
            r'-v\d+$',  # version at end only
        ]
        
        cleaned = full_name
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove training identifiers (animal names, swarm names, etc)
        # Pattern: -word_word_word at end
        cleaned = re.sub(r'-\w+_\w+_\w+$', '', cleaned)
        
        # Extract base: ModelName-Version-Variant (3 components)
        # Matches: Qwen2.5-0.5B-Instruct, Llama-3.1-8B, Mistral-7B-v0.1
        pattern = r'^([A-Za-z0-9]+(?:\.\d+)?-\d+(?:\.\d+)?[A-Z]?(?:-(?:Instruct|Base|Chat))?)'
        match = re.match(pattern, cleaned)
        
        if match:
            return match.group(1)
        
        # Fallback: first 3 dash-separated parts
        parts = cleaned.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:3])
        
        return cleaned[:50]
    
    def _generate_target_repo(self) -> str:
        """Auto-generate target repository"""
        
        manual_target = os.getenv("TARGET_REPO", "").strip()
        if manual_target:
            print(f"  🔧 Manual target repo: {manual_target}")
            return manual_target
        
        if self.upload_mode != "new_repo":
            return self.repo_id
        
        username = self.repo_id.split('/')[0]
        full_model_name = self.repo_id.split('/')[-1]
        base_name = self._extract_base_model_name(full_model_name)
        
        target = f"{username}/{base_name}-GGUF"
        print(f"  🤖 Auto-generated target repo: {target}")
        
        return target
    
    def _detect_base_model(self) -> Optional[str]:
        """Auto-detect base model"""
        manual = os.getenv("BASE_MODEL_TOKENIZER", "").strip()
        if manual:
            print(f"  🔧 Manual base model: {manual}")
            return manual
        
        try:
            from huggingface_hub import hf_hub_download
            
            print(f"  🔍 Auto-detecting base model...")
            config_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="config.json",
                token=self.hf_token,
                cache_dir=str(self.cache_dir)
            )
            
            with open(config_path) as f:
                config = json.load(f)
            
            arch = config.get("architectures", [None])[0]
            
            if arch in self.BASE_MODEL_MAPPING:
                base = self.BASE_MODEL_MAPPING[arch]
                print(f"  ✅ Auto-detected: {base}")
                return base
            else:
                print(f"  ⚠️  Unknown arch: {arch}")
        
        except Exception as e:
            print(f"  ⚠️  Auto-detect failed: {e}")
        
        print(f"     Using model's own tokenizer")
        return None
    
    @property
    def model_name(self) -> str:
        """Clean model name for filename"""
        full_name = self.repo_id.split("/")[-1]
        return self._extract_base_model_name(full_name)
    
    @property
    def convert_script(self) -> Path:
        return self.llama_cpp_path / "convert_hf_to_gguf.py"
    
    @property
    def quantize_binary(self) -> Path:
        return self.llama_cpp_path / "build/bin/llama-quantize"
    
    def get_output_filename(self, quant_type: str) -> str:
        return self.output_pattern.format(
            model_name=self.model_name,
            quant=quant_type
        )
    
    @property
    def valid_quants(self):
        return self.VALID_QUANTS
