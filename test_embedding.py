from transformers import AutoModel, AutoTokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

model = AutoModel.from_pretrained('models/Ace-Step1.5/Qwen3-Embedding-0.6B', trust_remote_code=True, torch_dtype=dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained('models/Ace-Step1.5/Qwen3-Embedding-0.6B', trust_remote_code=True)

print('Model loaded successfully')
print(f'Model device: {model.device}')
print(f'Model dtype: {model.dtype}')
