"""ACE-Step Gradio Client包装器 - 使用官方API"""
import os
import numpy as np
from typing import Optional
import soundfile as sf
from pathlib import Path

class ACEStepGradioWrapper:
    def __init__(self):
        self.client = None
        self.space_id = os.getenv("ACE_STEP_SPACE_ID", "ACE-Step/ACE-Step")
        self._init_client()
    
    def is_ready(self) -> bool:
        """检查API是否可用"""
        return self.client is not None
    
    def _init_client(self):
        """初始化Gradio客户端"""
        try:
            print("  连接ACE-Step官方API...")
            from gradio_client import Client
            self.client = Client(self.space_id)
            print("  ✓ ACE-Step API连接成功")
        except ImportError:
            print("  ⚠ 需要安装: pip install gradio_client")
            self.client = None
        except Exception as e:
            print(f"  ⚠ API连接失败: {e}")
            self.client = None
    
    def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        lyrics: str = None,
        seed: int = None
    ) -> Optional[np.ndarray]:
        """生成音乐"""
        if self.client is None:
            return None
        
        try:
            print(f"  使用ACE-Step API生成: {prompt}")

            # 不同Space的api_name / 参数名可能不同，这里做“多策略尝试”
            payload = {"duration": duration}
            if lyrics:
                payload["lyrics"] = lyrics
            if seed is not None:
                payload["seed"] = seed
            attempts = [
                ("/generate", {"prompt": prompt, **payload}),
                ("/generate", {"text": prompt, **payload}),
                ("/predict", {"prompt": prompt, **payload}),
                ("/predict", {"text": prompt, **payload}),
            ]
            last_err = None
            result = None
            for api_name, kwargs in attempts:
                try:
                    result = self.client.predict(api_name=api_name, **kwargs)
                    break
                except Exception as e:
                    last_err = e
                    result = None
            
            # result是音频文件路径
            if isinstance(result, str) and Path(result).exists():
                audio, sr = sf.read(result)
                print("  ✓ 音乐生成成功")
                return audio

            # 有些Space会返回 (path, ...) 或 dict
            if isinstance(result, (list, tuple)) and len(result) > 0:
                first = result[0]
                if isinstance(first, str) and Path(first).exists():
                    audio, sr = sf.read(first)
                    print("  ✓ 音乐生成成功")
                    return audio
            
            if last_err is not None:
                raise last_err
            return None
            
        except Exception as e:
            print(f"  生成失败: {e}")
            return None
