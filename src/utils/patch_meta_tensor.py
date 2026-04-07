"""修复vector_quantize_pytorch的meta tensor问题"""
import torch

# 保存原始的all方法
_original_all = torch.Tensor.all

def patched_all(self, *args, **kwargs):
    """修复meta tensor的all()调用"""
    if self.is_meta:
        # 对于meta tensor，假设条件为True
        return torch.tensor(True, device='cpu')
    return _original_all(self, *args, **kwargs)

# 应用补丁
torch.Tensor.all = patched_all

print("✓ vector_quantize_pytorch meta tensor补丁已应用")
