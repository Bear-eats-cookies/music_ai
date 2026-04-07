import os
import sys

os.environ['ACCELERATE_USE_META_DEVICE'] = '0'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import warnings
warnings.filterwarnings('ignore')

# 添加所有兼容补丁
try:
    import transformers.configuration_utils as _tf_cfg_utils
    if not hasattr(_tf_cfg_utils, "layer_type_validation"):
        def _layer_type_validation(layer_types):
            return layer_types
        _tf_cfg_utils.layer_type_validation = _layer_type_validation
except Exception:
    pass

try:
    import transformers.modeling_flash_attention_utils as _tf_fa_utils
    if not hasattr(_tf_fa_utils, "FlashAttentionKwargs"):
        _tf_fa_utils.FlashAttentionKwargs = dict
except Exception:
    pass

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

import torch

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

try:
    import accelerate.utils.memory as _acc_memory
    if not hasattr(_acc_memory, "clear_device_cache"):
        def _clear_device_cache(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _acc_memory.clear_device_cache = _clear_device_cache
except Exception as e:
    print(f"  ⚠ accelerate兼容补丁失败: {e}")

try:
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if 'qwen3' not in CONFIG_MAPPING:
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

# 现在导入wrapper
from src.music_generation.ace_step_wrapper import ACEStepWrapper

print("Creating ACEStepWrapper...")
ace = ACEStepWrapper("models/Ace-Step1.5")

print(f"Is ready: {ace.is_ready()}")
print(f"DiT model: {ace.dit_model is not None}")
print(f"VAE: {ace.vae is not None}")
print(f"Embedding: {ace.embedding_model is not None}")
print(f"Embedding tokenizer: {ace.emb_tokenizer is not None}")
