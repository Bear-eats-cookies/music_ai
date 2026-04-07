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
from diffusers import AutoencoderOobleck

print("Creating ACEStepWrapper...")
ace = ACEStepWrapper("models/Ace-Step1.5")

print(f"DiT model: {ace.dit_model is not None}")
print(f"VAE: {ace.vae is not None}")

# 加载真实的VAE
vae = AutoencoderOobleck.from_pretrained("models/Ace-Step1.5/vae", torch_dtype=torch.float32).to("cuda")
vae.eval()

# 生成一个音频latent
device = "cuda"
dtype = torch.float32
batch_size = 1
num_frames = 100

text_hidden = torch.randn(1, 10, 1024, device=device, dtype=dtype)
text_mask = torch.ones(1, 10, device=device, dtype=torch.long)
lyric_hidden = torch.randn(1, 10, 1024, device=device, dtype=dtype)
lyric_mask = torch.ones(1, 10, device=device, dtype=torch.long)
silence_latent = torch.randn(1, num_frames, 64, device=device, dtype=dtype)
refer_audio_hidden = silence_latent.clone()
refer_audio_mask = torch.tensor([0], device=device, dtype=torch.long)
src_latents = torch.randn(batch_size, num_frames, 64, device=device, dtype=dtype)
attention_mask = torch.ones(batch_size, num_frames, device=device, dtype=dtype)
chunk_masks = torch.ones(batch_size, num_frames, 64, device=device, dtype=dtype)
is_covers = torch.zeros(batch_size, device=device, dtype=torch.bool)
hidden_states_for_gen = silence_latent[:, :num_frames, :].expand(batch_size, -1, -1)

print("\nGenerating audio with DiT...")
with torch.no_grad():
    result = ace.dit_model.generate_audio(
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
        seed=42,
        fix_nfe=8,
        infer_method="ode",
        precomputed_lm_hints_25Hz=None,
        audio_cover_strength=0.0,
        non_cover_text_hidden_states=text_hidden,
        non_cover_text_attention_mask=text_mask
    )

if isinstance(result, dict):
    audio_latents = result.get("target_latents", src_latents)
elif isinstance(result, tuple):
    audio_latents = result[0]
else:
    audio_latents = result

print(f"Audio latents shape: {audio_latents.shape}")
print(f"Audio latents mean: {audio_latents.mean():.6f}")
print(f"Audio latents std: {audio_latents.std():.6f}")
print(f"Audio latents min: {audio_latents.min():.6f}")
print(f"Audio latents max: {audio_latents.max():.6f}")

# 用真实VAE解码
print("\nDecoding with real VAE...")
audio_latents_for_vae = audio_latents.transpose(1, 2)  # [B, T, C] -> [B, C, T]
decoded = vae.decode(audio_latents_for_vae)
audio = decoded.sample

print(f"Decoded audio shape: {audio.shape}")
print(f"Decoded audio mean: {audio.mean():.6f}")
print(f"Decoded audio std: {audio.std():.6f}")
print(f"Decoded audio min: {audio.min():.6f}")
print(f"Decoded audio max: {audio.max():.6f}")

# 保存音频
import soundfile as sf
audio_np = audio.detach().cpu().numpy().squeeze()
if audio_np.ndim == 2:
    audio_np = audio_np.transpose(1, 0)
sf.write("test_dit_output.wav", audio_np, 48000)
print("Saved to test_dit_output.wav")
