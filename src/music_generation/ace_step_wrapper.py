"""
ACE-Step模型包装器 - 完整版
支持: DiT + LM + VAE 全组件加载
"""
import os
import re
import sys

# 添加ACE-Step官方包路径
acestep_path = r"C:\Users\user\Desktop\ACE-Step-1.5"
if acestep_path not in sys.path:
    sys.path.insert(0, acestep_path)

# 在导入任何库之前设置环境变量
os.environ['ACCELERATE_USE_META_DEVICE'] = '0'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# 兼容补丁: layer_type_validation
try:
    import transformers.configuration_utils as _tf_cfg_utils
    if not hasattr(_tf_cfg_utils, "layer_type_validation"):
        def _layer_type_validation(layer_types):
            return layer_types
        _tf_cfg_utils.layer_type_validation = _layer_type_validation
except Exception:
    pass

# 兼容补丁: FlashAttentionKwargs
try:
    import transformers.modeling_flash_attention_utils as _tf_fa_utils
    if not hasattr(_tf_fa_utils, "FlashAttentionKwargs"):
        _tf_fa_utils.FlashAttentionKwargs = dict
except Exception:
    pass

# 兼容补丁: transformers.utils 缺少的新装饰器
try:
    import transformers.utils as _tf_utils

    def _noop_decorator(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _wrapper(obj):
            return obj

        return _wrapper

    if not hasattr(_tf_utils, "auto_docstring"):
        _tf_utils.auto_docstring = _noop_decorator
    if not hasattr(_tf_utils, "can_return_tuple"):
        _tf_utils.can_return_tuple = _noop_decorator
except Exception:
    pass

# 兼容补丁: transformers.modeling_layers (新版本transformers移除了这个模块)
try:
    import transformers
    if not hasattr(transformers, 'modeling_layers'):
        import types
        modeling_layers = types.ModuleType('modeling_layers')
        class GradientCheckpointingLayer(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        modeling_layers.GradientCheckpointingLayer = GradientCheckpointingLayer
        transformers.modeling_layers = modeling_layers
        sys.modules['transformers.modeling_layers'] = modeling_layers
except Exception as e:
    print(f"  ⚠ modeling_layers补丁失败: {e}")

# 兼容补丁: diffusers 依赖的新版本 accelerate API
try:
    import accelerate.utils.memory as _acc_memory
    if not hasattr(_acc_memory, "clear_device_cache"):
        def _clear_device_cache(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _acc_memory.clear_device_cache = _clear_device_cache
except Exception as e:
    print(f"  ⚠ accelerate兼容补丁失败: {e}")

# 兼容补丁: Qwen3模型支持
try:
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    
    # 检查是否已支持Qwen3
    if 'qwen3' not in CONFIG_MAPPING:
        # 尝试使用Qwen2作为替代
        try:
            from transformers.models.qwen2 import Qwen2Config
            CONFIG_MAPPING.register('qwen3', Qwen2Config)
            print("  ✓ Qwen3配置已注册(使用Qwen2兼容模式)")
        except Exception:
            pass
except Exception as e:
    print(f"  ⚠ Qwen3兼容补丁失败: {e}")

try:
    import types
    import transformers.models.qwen2.modeling_qwen2 as _qwen2_modeling
    if "transformers.models.qwen3" not in sys.modules:
        sys.modules["transformers.models.qwen3"] = types.ModuleType("transformers.models.qwen3")
    qwen3_modeling = types.ModuleType("transformers.models.qwen3.modeling_qwen3")
    qwen3_modeling.Qwen3MLP = _qwen2_modeling.Qwen2MLP
    qwen3_modeling.Qwen3RMSNorm = _qwen2_modeling.Qwen2RMSNorm
    qwen3_modeling.Qwen3RotaryEmbedding = _qwen2_modeling.Qwen2RotaryEmbedding
    qwen3_modeling.apply_rotary_pos_emb = _qwen2_modeling.apply_rotary_pos_emb
    qwen3_modeling.eager_attention_forward = _qwen2_modeling.eager_attention_forward
    sys.modules["transformers.models.qwen3.modeling_qwen3"] = qwen3_modeling
except Exception as e:
    print(f"  ⚠ Qwen3 modeling兼容补丁失败: {e}")

# Monkey patch修复vector_quantize_pytorch的meta tensor问题
original_all = torch.Tensor.all
def patched_all(self, *args, **kwargs):
    if self.is_meta:
        return torch.tensor(True, device='cpu')
    return original_all(self, *args, **kwargs)
torch.Tensor.all = patched_all


class ACEStepWrapper:
    """ACE-Step v1.5 Turbo 完整包装器"""
    
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 使用float32以避免bfloat16在某些操作上的兼容性问题
        # (如interpolate、one_hot矩阵乘法等)
        self.dtype = torch.float32
        self.model_path = Path(model_path)
        
        # 模型组件
        self.dit_model = None
        self.lm_model = None
        self.vae = None
        self.tokenizer = None
        self.embedding_model = None
        self.emb_tokenizer = None
        self.silence_latent = None
        self.load_lm = os.getenv("ACE_STEP_LOAD_LM", "0") == "1"
        self.allow_fake_components = os.getenv("ACE_STEP_ALLOW_FAKE_COMPONENTS", "0") == "1"
        self.require_official_backend = os.getenv("ACE_STEP_REQUIRE_OFFICIAL_BACKEND", "1") == "1"
        self.prefer_official_backend = (
            self.require_official_backend
            or os.getenv("ACE_STEP_PREFER_OFFICIAL_BACKEND", "1") == "1"
        )
        self.official_backend = None
        self.official_backend_name = None
        self.official_backend_error = None
        
        if self.model_path.exists():
            if self.prefer_official_backend:
                self._init_official_backend()
            self._load_all_models()

    def _init_official_backend(self):
        """Try to initialize the official ACE-Step inference backend if installed."""
        try:
            from acestep.handler import AceStepHandler
            from acestep.inference import GenerationConfig, GenerationParams, format_sample, generate_music
            from acestep.llm_inference import LLMHandler
        except Exception:
            try:
                from acestep.handler.base import GenerationConfig, GenerationParams
                from acestep.handler.dit_handler import AceStepHandler
                from acestep.handler.llm_handler import LLMHandler
                from acestep.inference import format_sample, generate_music
            except Exception as e:
                self.official_backend_error = f"官方 ACE-Step 推理包不可用: {e}"
                print(f"  ⚠ {self.official_backend_error}")
                return

        dit_handler = AceStepHandler()
        try:
            dit_handler.initialize_service(
                project_root=str(self.model_path),
                config_path="acestep-v15-turbo",
                device=self.device,
            )
        except Exception as e:
            self.official_backend_error = f"官方 ACE-Step DiT 初始化失败: {e}"
            print(f"  ⚠ {self.official_backend_error}")
            return

        llm_handler = None
        llm_backend_candidates = []
        explicit_backend = os.getenv("ACE_STEP_LLM_BACKEND")
        if explicit_backend:
            llm_backend_candidates.append(explicit_backend)
        llm_backend_candidates.extend(["transformers", "vllm"])

        if self.load_lm:
            for backend_name in llm_backend_candidates:
                try:
                    llm_handler = LLMHandler()
                    llm_handler.initialize(
                        checkpoint_dir=str(self.model_path),
                        lm_model_path="acestep-5Hz-lm-1.7B",
                        backend=backend_name,
                        device=self.device,
                    )
                    self.official_backend_name = f"official+{backend_name}"
                    break
                except Exception as e:
                    llm_handler = None
                    self.official_backend_error = f"官方 ACE-Step LM 初始化失败({backend_name}): {e}"
            if llm_handler is None and self.official_backend_error:
                print(f"  ⚠ {self.official_backend_error}")

        self.official_backend = {
            "GenerationConfig": GenerationConfig,
            "GenerationParams": GenerationParams,
            "format_sample": format_sample,
            "generate_music": generate_music,
            "dit_handler": dit_handler,
            "llm_handler": llm_handler,
        }
        if self.official_backend_name is None:
            self.official_backend_name = "official"
        print(f"  ✓ 官方 ACE-Step 推理后端已就绪: {self.official_backend_name}")
    
    def _load_all_models(self):
        """加载所有模型组件"""
        if self.official_backend is not None:
            print("=" * 50)
            print("检测到官方 ACE-Step 推理后端，跳过轻量本地包装器加载。")
            print(f"  模型路径: {self.model_path}")
            print(f"  后端: {self.official_backend_name}")
            print("=" * 50)
            return

        if self.require_official_backend:
            print("=" * 50)
            print("ACE-Step official backend is required; skipping local wrapper load.")
            if self.official_backend_error:
                print(f"  Error: {self.official_backend_error}")
            else:
                print("  Error: official backend was not initialized")
            print("=" * 50)
            return

        print("=" * 50)
        print("正在加载ACE-Step v1.5 Turbo模型...")
        print(f"  模型路径: {self.model_path}")
        print(f"  设备: {self.device}, 数据类型: {self.dtype}")
        print("=" * 50)
        
        self._load_dit_model()
        if self.load_lm:
            self._load_lm_model()
        else:
            print("\n[2/5] 跳过LM模型加载 (当前生成链路未使用，可通过 ACE_STEP_LOAD_LM=1 启用)")
        self._load_vae()
        self._load_embedding_model()
        self._load_silence_latent()
        
        print("\n" + "=" * 50)
        print("模型加载状态:")
        print(f"  DiT模型: {'✓ 已加载' if self.dit_model else '✗ 未加载'}")
        print(f"  LM模型: {'✓ 已加载' if self.lm_model else '✗ 未加载'}")
        print(f"  VAE: {'✓ 已加载' if self.vae else '✗ 未加载'}")
        print(f"  Embedding: {'✓ 已加载' if self.embedding_model else '✗ 未加载'}")
        print("=" * 50)
    
    def _load_dit_model(self):
        """加载DiT生成模型"""
        try:
            dit_path = self.model_path / "acestep-v15-turbo"
            if not dit_path.exists():
                print(f"  ⚠ DiT模型路径不存在: {dit_path}")
                return
            
            print(f"\n[1/5] 加载DiT模型...")
            
            dit_path_str = str(dit_path.resolve())
            if dit_path_str not in sys.path:
                sys.path.insert(0, dit_path_str)
            
            # 方法1: 使用AutoModel
            try:
                from transformers import AutoModel
                self.dit_model = AutoModel.from_pretrained(
                    str(dit_path),
                    trust_remote_code=True,
                    torch_dtype=self.dtype
                ).to(self.device)
                
                self.dit_model.eval()
                print(f"  ✓ DiT模型加载成功 (AutoModel)")
                return
            except Exception as e:
                print(f"  ⚠ AutoModel加载失败: {e}")
            
            # 方法2: 直接导入本地模型文件
            try:
                import importlib.util
                model_file = dit_path / "modeling_acestep_v15_turbo.py"
                if model_file.exists():
                    # 先注册模块
                    spec = importlib.util.spec_from_file_location("modeling_acestep_v15_turbo", str(model_file))
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["modeling_acestep_v15_turbo"] = module
                    spec.loader.exec_module(module)
                    
                    # 查找模型类
                    model_class = None
                    for name in dir(module):
                        obj = getattr(module, name)
                        if isinstance(obj, type) and name.endswith('Model'):
                            model_class = obj
                            break
                    
                    if model_class:
                        self.dit_model = model_class.from_pretrained(
                            str(dit_path), 
                            torch_dtype=self.dtype
                        ).to(self.device)
                        self.dit_model.eval()
                        print(f"  ✓ DiT模型加载成功 (本地导入)")
                        return
                    else:
                        print(f"  ⚠ 未找到模型类")
            except Exception as e2:
                print(f"  ⚠ 本地导入失败: {e2}")
            
            if self.allow_fake_components:
                # 仅用于调试。该路径会退化为伪造组件，不能保证生成真实音乐。
                try:
                    print("  尝试手动加载DiT权重...")
                    self.dit_model = self._load_dit_manual(dit_path)
                    if self.dit_model:
                        print(f"  ⚠ DiT仅以调试模式加载成功 (手动加载)")
                        return
                except Exception as e3:
                    print(f"  ⚠ 手动加载失败: {e3}")
            
            print("  ⚠ DiT模型所有加载方法均失败")
            
        except Exception as e:
            print(f"  ⚠ DiT模型加载失败: {e}")
            self.dit_model = None
    
    def _load_dit_manual(self, dit_path):
        """手动加载DiT模型权重"""
        import json
        from safetensors.torch import load_file
        
        # 加载配置
        config_file = dit_path / "config.json"
        if not config_file.exists():
            return None
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 加载权重
        weight_files = list(dit_path.glob("*.safetensors")) + list(dit_path.glob("*.bin"))
        if not weight_files:
            return None
        
        state_dict = {}
        for wf in weight_files:
            if wf.suffix == '.safetensors':
                state_dict.update(load_file(str(wf)))
            else:
                state_dict.update(torch.load(str(wf), map_location='cpu', weights_only=False))
        
        # 创建简单的模型包装器
        return SimpleDiTWrapper(state_dict, config, self.device, self.dtype)
    
    def _load_lm_model(self):
        """加载语言模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            lm_path = self.model_path / "acestep-5Hz-lm-1.7B"
            if not lm_path.exists():
                print(f"  ⚠ LM模型路径不存在: {lm_path}")
                return
            
            print(f"\n[2/5] 加载LM模型 (Qwen3)...")
            
            try:
                self.lm_model = AutoModelForCausalLM.from_pretrained(
                    str(lm_path),
                    trust_remote_code=True,
                    torch_dtype=self.dtype
                ).to(self.device)
                
                self.lm_model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(str(lm_path), trust_remote_code=True)
                print(f"  ✓ LM模型加载成功")
            except Exception as e:
                print(f"  ⚠ LM模型加载失败(将跳过): {e}")
                # LM模型是可选的，不影响核心生成功能
                self.lm_model = None
                self.tokenizer = None
            
        except Exception as e:
            print(f"  ⚠ LM模型加载失败: {e}")
            self.lm_model = None
    
    def _load_vae(self):
        """加载VAE解码器"""
        try:
            vae_path = self.model_path / "vae"
            if not vae_path.exists():
                print(f"  ⚠ VAE路径不存在: {vae_path}")
                return
            
            print(f"\n[3/5] 加载VAE解码器...")
            
            try:
                from diffusers import AutoencoderOobleck
                self.vae = AutoencoderOobleck.from_pretrained(str(vae_path), torch_dtype=self.dtype).to(self.device)
                self.vae.eval()
                print(f"  ✓ VAE加载成功")
            except ImportError as e:
                print(f"  ⚠ diffusers导入失败: {e}")
                if self.allow_fake_components:
                    self._load_vae_manual(vae_path)
            except Exception as e:
                print(f"  ⚠ VAE加载失败: {e}")
                if self.allow_fake_components:
                    self._load_vae_manual(vae_path)
            
        except Exception as e:
            print(f"  ⚠ VAE加载失败: {e}")
            self.vae = None
    
    def _load_vae_manual(self, vae_path):
        """手动加载VAE"""
        try:
            import json
            from safetensors.torch import load_file
            
            config_file = vae_path / "config.json"
            model_file = vae_path / "diffusion_pytorch_model.safetensors"
            
            if not config_file.exists() or not model_file.exists():
                return
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            state_dict = load_file(str(model_file))
            self.vae = SimpleVAEWrapper(state_dict, config, self.device, self.dtype)
            print(f"  ✓ VAE手动加载成功")
            
        except Exception as e:
            print(f"  ⚠ VAE手动加载失败: {e}")
            self.vae = None
    
    def _load_embedding_model(self):
        """加载文本嵌入模型"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            emb_path = self.model_path / "Qwen3-Embedding-0.6B"
            if not emb_path.exists():
                return
            
            print(f"\n[4/5] 加载Embedding模型...")
            
            try:
                self.embedding_model = AutoModel.from_pretrained(
                    str(emb_path), trust_remote_code=True, torch_dtype=self.dtype
                ).to(self.device)
                
                self.embedding_model.eval()
                self.emb_tokenizer = AutoTokenizer.from_pretrained(str(emb_path), trust_remote_code=True)
                print(f"  ✓ Embedding模型加载成功")
            except Exception as e:
                print(f"  ⚠ Embedding模型加载失败(将跳过): {e}")
                # Embedding模型是可选的，会使用随机嵌入
                self.embedding_model = None
                self.emb_tokenizer = None
            
        except Exception as e:
            print(f"  ⚠ Embedding模型加载失败: {e}")
            self.embedding_model = None
    
    def _load_silence_latent(self):
        """加载静音潜变量"""
        try:
            silence_path = self.model_path / "acestep-v15-turbo" / "silence_latent.pt"
            if silence_path.exists():
                print(f"\n[5/5] 加载静音潜变量...")
                self.silence_latent = torch.load(str(silence_path), map_location=self.device, weights_only=True)
                if self.silence_latent.dtype != self.dtype:
                    self.silence_latent = self.silence_latent.to(self.dtype)
                # 调整形状为 [1, T, 64]
                if self.silence_latent.shape[1] == 64:
                    self.silence_latent = self.silence_latent.transpose(1, 2)
                print(f"  ✓ 静音潜变量加载成功, shape: {self.silence_latent.shape}")
        except Exception as e:
            print(f"  ⚠ 静音潜变量加载失败: {e}")
            self.silence_latent = None
    
    def is_ready(self) -> bool:
        """检查是否具备完整的生成能力"""
        if self.require_official_backend:
            return self.official_backend is not None

        if self.official_backend is not None:
            return True

        required_components = [
            self.dit_model is not None,
            self.vae is not None,
            self.embedding_model is not None,
            self.emb_tokenizer is not None,
        ]
        fake_components = (
            isinstance(self.dit_model, SimpleDiTWrapper)
            or isinstance(self.vae, SimpleVAEWrapper)
        )
        return all(required_components) and not fake_components

    def get_backend_status(self) -> Dict[str, Any]:
        """Return a compact status summary for the active ACE-Step backend."""
        local_wrapper_ready = False
        if not self.require_official_backend:
            required_components = [
                self.dit_model is not None,
                self.vae is not None,
                self.embedding_model is not None,
                self.emb_tokenizer is not None,
            ]
            fake_components = (
                isinstance(self.dit_model, SimpleDiTWrapper)
                or isinstance(self.vae, SimpleVAEWrapper)
            )
            local_wrapper_ready = all(required_components) and not fake_components

        if self.official_backend is not None:
            selected_backend = self.official_backend_name or "official"
        elif self.require_official_backend:
            selected_backend = "official_required_unavailable"
        elif local_wrapper_ready:
            selected_backend = "local_wrapper"
        else:
            selected_backend = "unavailable"

        return {
            "selected_backend": selected_backend,
            "official_backend_ready": self.official_backend is not None,
            "official_backend_name": self.official_backend_name,
            "official_backend_error": self.official_backend_error,
            "require_official_backend": self.require_official_backend,
            "prefer_official_backend": self.prefer_official_backend,
            "local_wrapper_ready": local_wrapper_ready,
            "allow_fake_components": self.allow_fake_components,
            "load_lm": self.load_lm,
        }
    
    def _encode_text(self, text: str) -> tuple:
        """编码文本提示词"""
        text_hidden_dim = 1024
        
        if self.embedding_model is not None and self.emb_tokenizer is not None:
            try:
                inputs = self.emb_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    hidden_states = outputs.last_hidden_state.to(self.dtype)
                # attention_mask保持long类型用于索引
                return hidden_states, inputs.attention_mask
            except Exception as e:
                print(f"    文本编码失败: {e}")
        
        seq_len = min(len(text.split()), 50) + 2
        hidden_states = torch.randn(1, seq_len, text_hidden_dim, device=self.device, dtype=self.dtype)
        attention_mask = torch.ones(1, seq_len, device=self.device, dtype=torch.long)
        return hidden_states, attention_mask
    
    def _encode_lyrics(self, lyrics: str) -> tuple:
        """编码歌词"""
        lyric_hidden_dim = 1024
        
        try:
            if self.embedding_model is not None and self.emb_tokenizer is not None:
                emb_inputs = self.emb_tokenizer(lyrics, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
                with torch.no_grad():
                    outputs = self.embedding_model(**emb_inputs)
                    hidden_states = outputs.last_hidden_state.to(self.dtype)
                return hidden_states, emb_inputs.attention_mask
        except Exception as e:
            print(f"    歌词编码失败: {e}")
        
        seq_len = min(len(lyrics), 100)
        hidden_states = torch.randn(1, seq_len, lyric_hidden_dim, device=self.device, dtype=self.dtype)
        attention_mask = torch.ones(1, seq_len, device=self.device, dtype=torch.long)
        return hidden_states, attention_mask
    
    def _generate_lm_hints(self, prompt: str, target_length: int = 512) -> torch.Tensor:
        """使用LM生成音乐结构提示 - 返回None让模型自动处理"""
        return None

    def _infer_vocal_language(self, lyrics: Optional[str]) -> str:
        """Infer ACE-Step vocal language from lyric text."""
        if not lyrics:
            return "en"

        text = lyrics.strip()
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[\u3040-\u30ff]", text):
            return "ja"
        if re.search(r"[\uac00-\ud7af]", text):
            return "ko"
        return "en"

    def _normalize_lyrics(self, lyrics: Optional[str]) -> Optional[str]:
        """Normalize user lyrics into the structured format ACE-Step expects."""
        if not lyrics:
            return None

        text = lyrics.replace("\r\n", "\n").strip()
        if not text:
            return None

        if "[" in text and "]" in text:
            return text

        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        if not blocks:
            return None

        structured_blocks = []
        for index, block in enumerate(blocks):
            tag = "[Verse]" if index == 0 else "[Chorus]"
            structured_blocks.append(f"{tag}\n{block}")
        return "\n\n".join(structured_blocks)

    def _generate_music_with_official_backend(
        self,
        prompt: str,
        duration: int,
        lyrics: Optional[str],
        seed: Optional[int],
        vocal_language: Optional[str],
    ) -> Optional[np.ndarray]:
        """Generate music via the official ACE-Step inference package when available."""
        backend = self.official_backend
        if backend is None:
            return None

        normalized_lyrics = self._normalize_lyrics(lyrics)
        inferred_language = vocal_language or self._infer_vocal_language(normalized_lyrics)
        instrumental = not bool(normalized_lyrics)
        prompt_caption = prompt.strip()
        if not instrumental and "sing" not in prompt_caption.lower() and "vocal" not in prompt_caption.lower():
            prompt_caption = f"{prompt_caption}, expressive lead vocal singing the provided lyrics"
        requested_duration = -1.0 if normalized_lyrics else float(duration)

        if backend["llm_handler"] is not None and backend.get("format_sample") is not None:
            try:
                sample = backend["format_sample"](
                    llm_handler=backend["llm_handler"],
                    caption=prompt_caption,
                    lyrics=normalized_lyrics,
                    user_metadata={"language": inferred_language},
                )
                if getattr(sample, "success", False):
                    prompt_caption = getattr(sample, "caption", prompt_caption)
                    normalized_lyrics = getattr(sample, "lyrics", normalized_lyrics)
                    requested_duration = float(getattr(sample, "duration", requested_duration) or requested_duration)
                    inferred_language = getattr(sample, "language", inferred_language) or inferred_language
            except Exception as e:
                print(f"  ⚠ 官方格式化歌词/提示词失败，将直接使用原始输入: {e}")

        params = backend["GenerationParams"](
            task_type="text2music",
            caption=prompt_caption,
            lyrics=normalized_lyrics or "[Instrumental]",
            duration=requested_duration,
            instrumental=instrumental,
            vocal_language=inferred_language,
            inference_steps=8,
            seed=seed if seed is not None else -1,
            infer_method="ode",
            shift=3.0,
            use_cot_metas=backend["llm_handler"] is not None,
            use_cot_caption=backend["llm_handler"] is not None,
            use_cot_lyrics=backend["llm_handler"] is not None and not instrumental,
            use_cot_language=backend["llm_handler"] is not None,
            thinking=backend["llm_handler"] is not None,
        )

        seeds = [seed] if seed is not None else None
        config = backend["GenerationConfig"](
            batch_size=1,
            use_random_seed=seed is None,
            seeds=seeds,
            audio_format="wav",
        )

        try:
            result = backend["generate_music"](
                backend["dit_handler"],
                backend["llm_handler"],
                params,
                config,
                save_dir=None,
            )
        except Exception as e:
            self.official_backend_error = f"官方 ACE-Step 推理失败: {e}"
            print(f"  ⚠ {self.official_backend_error}")
            return None

        if not getattr(result, "success", False):
            self.official_backend_error = f"官方 ACE-Step 推理失败: {getattr(result, 'error', 'unknown error')}"
            print(f"  ⚠ {self.official_backend_error}")
            return None

        audios = getattr(result, "audios", None) or []
        if not audios:
            self.official_backend_error = "官方 ACE-Step 未返回音频"
            print(f"  ⚠ {self.official_backend_error}")
            return None

        first_audio = audios[0]
        tensor = first_audio.get("tensor") if isinstance(first_audio, dict) else None
        if tensor is not None:
            if isinstance(tensor, torch.Tensor):
                audio = tensor.detach().cpu().numpy()
            else:
                audio = np.asarray(tensor)
            audio = np.squeeze(audio)
            if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
                audio = audio.transpose(1, 0)
            return audio.astype(np.float32)

        path_value = first_audio.get("path") if isinstance(first_audio, dict) else None
        if path_value and Path(path_value).exists():
            audio, _ = sf.read(path_value, dtype="float32")
            return np.asarray(audio, dtype=np.float32)

        self.official_backend_error = "官方 ACE-Step 返回结果中缺少可读取的音频"
        print(f"  ⚠ {self.official_backend_error}")
        return None

    def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        lyrics: str = None,
        seed: int = None,
        vocal_language: str = None,
    ) -> Optional[np.ndarray]:
        """生成音乐"""
        if self.official_backend is not None:
            print("\n开始生成音乐（官方 ACE-Step 后端）...")
            print(f"  提示词: {prompt}")
            print(f"  时长: {duration}秒")
            audio = self._generate_music_with_official_backend(
                prompt=prompt,
                duration=duration,
                lyrics=lyrics,
                seed=seed,
                vocal_language=vocal_language,
            )
            if audio is not None:
                print(f"  ✓ 官方 ACE-Step 生成成功! 音频长度: {len(audio)} 样本")
                return audio
            print("  ⚠ 官方后端失败，回退到当前本地包装器。")
        elif self.prefer_official_backend:
            print("  ⚠ 当前未接入官方 ACE-Step 推理后端，将使用本地轻量包装器；这一路径不如官方推理稳定。")

        if not self.is_ready():
            print("  ⚠ 模型未完全加载，无法生成")
            return None
        
        try:
            print(f"\n开始生成音乐...")
            print(f"  提示词: {prompt}")
            print(f"  时长: {duration}秒")
            
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            sample_rate = 48000
            vae_downsample = 1920
            num_frames = int(duration * sample_rate / vae_downsample)
            num_frames = max(num_frames, 64)
            
            print(f"  目标帧数: {num_frames}")
            
            # 编码文本
            print("  [1/4] 编码文本提示...")
            text_hidden, text_mask = self._encode_text(prompt)
            
            # 编码歌词
            print("  [2/4] 编码歌词...")
            normalized_lyrics = self._normalize_lyrics(lyrics)
            if normalized_lyrics:
                lyric_hidden, lyric_mask = self._encode_lyrics(normalized_lyrics)
            else:
                lyric_hidden = torch.randn(1, 10, 1024, device=self.device, dtype=self.dtype)
                lyric_mask = torch.ones(1, 10, device=self.device, dtype=torch.long)
            
            # 生成LM提示
            print("  [3/4] 生成结构提示...")
            lm_hints = self._generate_lm_hints(prompt, target_length=256)
            
            # 准备生成参数
            print("  [4/4] 执行DiT生成...")
            
            batch_size = 1
            
            # 准备silence_latent
            if self.silence_latent is not None:
                silence_latent = self.silence_latent
                if silence_latent.shape[1] < num_frames:
                    repeat_times = (num_frames // silence_latent.shape[1]) + 1
                    silence_latent = silence_latent.repeat(1, repeat_times, 1)[:, :num_frames, :]
            else:
                silence_latent = torch.zeros(1, num_frames, 64, device=self.device, dtype=self.dtype)
            
            # 参考音频特征 - 使用silence_latent作为虚拟参考
            # refer_audio_order_mask: 指示每个参考音频属于哪个批次
            # 格式: torch.LongTensor([0, 0, 1]) 表示3个参考音频，前2个属批次0，第3个属批次1
            # 对于text2music模式(无参考音频)，我们使用silence_latent作为虚拟参考
            # 需要为每个批次提供一个参考音频(即使是虚拟的)
            refer_audio_hidden = silence_latent.clone().to(self.dtype)  # [1, T, 64] - 打包的参考音频特征
            # refer_audio_order_mask: 每个参考音频对应的批次索引 (必须是Long类型用于索引)
            # 对于batch_size=1，只有一个参考音频，索引为0
            refer_audio_mask = torch.tensor([0], device=self.device, dtype=torch.long)  # [N] - N个参考音频的批次索引
            
            # 其他参数
            src_latents = torch.randn(batch_size, num_frames, 64, device=self.device, dtype=self.dtype)
            attention_mask = torch.ones(batch_size, num_frames, device=self.device, dtype=self.dtype)
            # chunk_masks: [B, T, 64] - 用于标记音频块
            chunk_masks = torch.ones(batch_size, num_frames, 64, device=self.device, dtype=self.dtype)
            is_covers = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            hidden_states_for_gen = silence_latent[:, :num_frames, :].expand(batch_size, -1, -1)
            
            # 调用DiT生成
            with torch.no_grad():
                if hasattr(self.dit_model, 'generate_audio'):
                    result = self.dit_model.generate_audio(
                        text_hidden_states=text_hidden,
                        text_attention_mask=text_mask,
                        lyric_hidden_states=lyric_hidden,
                        lyric_attention_mask=lyric_mask,
                        refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                        refer_audio_order_mask=refer_audio_mask,
                        src_latents=hidden_states_for_gen,
                        chunk_masks=chunk_masks,
                        is_covers=is_covers,
                        silence_latent=silence_latent,
                        attention_mask=attention_mask,
                        seed=seed,
                        fix_nfe=8,
                        infer_method="ode",
                        precomputed_lm_hints_25Hz=lm_hints,
                        audio_cover_strength=0.0,
                        # 对于text2music模式，需要提供non_cover参数
                        non_cover_text_hidden_states=text_hidden,
                        non_cover_text_attention_mask=text_mask
                    )
                    
                    if isinstance(result, torch.Tensor):
                        audio_latents = result
                    elif isinstance(result, dict):
                        # generate_audio返回字典: {"target_latents": x_gen, "time_costs": time_costs}
                        audio_latents = result.get("target_latents", src_latents)
                        time_costs = result.get("time_costs", {})
                        if time_costs:
                            print(f"  生成耗时: {time_costs.get('total_time_cost', 0):.2f}秒")
                    elif isinstance(result, tuple):
                        audio_latents = result[0]
                    else:
                        audio_latents = src_latents
                else:
                    print("  ⚠ DiT模型没有generate_audio方法")
                    return None
            
            # VAE解码
            print("  解码音频...")
            with torch.no_grad():
                if hasattr(self.vae, 'decode'):
                    # VAE期望输入格式: [B, C, T] (批次, 通道, 时间)
                    # 当前audio_latents格式: [B, T, C]
                    print(f"  DiT输出形状: {audio_latents.shape}")
                    
                    # 转换为VAE期望的格式
                    audio_latents_for_vae = audio_latents.transpose(1, 2)  # [B, T, C] -> [B, C, T]
                    print(f"  VAE输入形状: {audio_latents_for_vae.shape}")
                    
                    decoded = self.vae.decode(audio_latents_for_vae)
                    
                    # VAE返回的是OobleckDecoderOutput对象，需要提取sample属性
                    if hasattr(decoded, 'sample'):
                        audio = decoded.sample
                        print(f"  VAE输出形状: {audio.shape}")
                    else:
                        audio = decoded
                        print(f"  VAE输出类型: {type(audio)}")
                else:
                    print("  ⚠ VAE没有decode方法")
                    return None
            
            audio = audio.cpu().numpy().squeeze()
            
            if audio.ndim == 0:
                audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
            elif audio.ndim == 2:
                # diffusers VAE通常输出 [channels, samples]，soundfile 需要 [samples, channels]
                if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
                    audio = audio.transpose(1, 0)
            elif audio.ndim > 2:
                audio = audio.reshape(audio.shape[-1])
            
            print(f"  ✓ 音乐生成成功! 音频长度: {len(audio)} 样本")
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"  ⚠ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class SimpleVAEWrapper:
    """简单的VAE包装器"""
    
    def __init__(self, state_dict: dict, config: dict, device: str, dtype: torch.dtype):
        self.state_dict = state_dict
        self.config = config
        self.device = device
        self.dtype = dtype
        self.sampling_rate = config.get('sampling_rate', 48000)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """简化的解码方法 - 改进版v2"""
        # interpolate不支持bfloat16，转换为float32
        latents = latents.float()
        
        # latents: [B, T, 64] - 需要转换为音频波形
        if latents.dim() == 4:
            latents = latents.squeeze(1)
            if latents.dim() == 4:
                latents = latents.permute(0, 2, 1, 3).reshape(latents.shape[0], -1, latents.shape[3])
        
        if latents.dim() == 3:
            if latents.shape[1] == 64:
                latents = latents.transpose(1, 2)
        
        B, T, C = latents.shape  # [B, T, 64]
        
        # VAE解码的关键：将latent转换为音频
        # 使用转置卷积风格的方法，而不是简单的插值
        
        upsample_factor = 1920  # 48000 / 25 = 1920
        
        # 方法：将latent视为频谱，生成多频率混合的音频
        # 每个通道对应不同的频率成分
        
        audio = torch.zeros(B, T * upsample_factor, device=latents.device, dtype=latents.dtype)
        
        # 使用latent值作为不同频率的振幅
        # 基频范围：20Hz - 2000Hz
        base_freq = 20.0
        freq_range = 2000.0 - base_freq
        
        # 创建时间轴
        t = torch.linspace(0, T * upsample_factor / 48000, T * upsample_factor, device=latents.device, dtype=latents.dtype)
        
        # 使用前16个通道生成不同频率的正弦波
        for i in range(min(C, 16)):
            # 每个通道对应一个频率
            freq = base_freq + (freq_range * i / 16)
            
            # 获取该通道的latent值作为振幅包络
            channel_latent = latents[:, :, i]  # [B, T]
            
            # 上采样振幅包络 - 使用更平滑的方法
            # 先用更大的kernel进行插值
            amplitude = torch.nn.functional.interpolate(
                channel_latent.unsqueeze(1),
                size=T * upsample_factor,
                mode='linear',
                align_corners=True
            ).squeeze(1)
            
            # 生成正弦波并乘以振幅包络
            wave = torch.sin(2 * 3.14159 * freq * t) * amplitude
            
            # 添加权重（低频权重高，高频权重低）
            weight = 1.0 / (1 + i * 0.5)
            audio = audio + wave * weight * 0.1
        
        # 添加噪声以增加变化
        noise = torch.randn_like(audio) * 0.02
        audio = audio + noise
        
        # 归一化
        audio = audio / (audio.abs().max() + 1e-8)
        
        return audio
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        return self


class SimpleDiTWrapper:
    """简单的DiT模型包装器 - 用于手动加载"""
    
    def __init__(self, state_dict: dict, config: dict, device: str, dtype: torch.dtype):
        self.state_dict = state_dict
        self.config = config
        self.device = device
        self.dtype = dtype
        self.hidden_size = config.get('hidden_size', 1024)
    
    def generate_audio(
        self,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.Tensor,
        refer_audio_order_mask: torch.Tensor,
        src_latents: torch.Tensor,
        chunk_masks: torch.Tensor,
        is_covers: torch.Tensor,
        silence_latent: torch.Tensor,
        attention_mask: torch.Tensor,
        seed: int = None,
        fix_nfe: int = 8,
        infer_method: str = "ode",
        precomputed_lm_hints_25Hz: torch.Tensor = None,
        audio_cover_strength: float = 0.0,
        non_cover_text_hidden_states: torch.Tensor = None,
        non_cover_text_attention_mask: torch.Tensor = None,
        **kwargs
    ) -> dict:
        """
        简化的音频生成方法
        由于没有完整的模型架构，这里生成随机的latent并返回
        """
        batch_size = src_latents.shape[0]
        num_frames = src_latents.shape[1]
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
        
        # 生成随机的音频latent (模拟DiT输出)
        target_latents = torch.randn(
            batch_size, num_frames, 64,
            device=self.device,
            dtype=self.dtype
        )
        
        # 添加一些基于输入的变化
        if text_hidden_states is not None:
            target_latents = target_latents + text_hidden_states.mean() * 0.1
        
        return {
            "target_latents": target_latents,
            "time_costs": {"total_time_cost": 0.5}
        }
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        return self
