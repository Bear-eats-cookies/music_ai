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

import torch
from transformers import AutoModel, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

emb_path = 'models/Ace-Step1.5/Qwen3-Embedding-0.6B'
print(f"Loading embedding model from {emb_path}...")

model = AutoModel.from_pretrained(emb_path, trust_remote_code=True, torch_dtype=dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained(emb_path, trust_remote_code=True)

print('Model loaded successfully')
print(f'Model device: {model.device}')
print(f'Model dtype: {model.dtype}')
