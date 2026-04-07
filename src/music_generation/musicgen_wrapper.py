"""MusicGen包装器"""

class MusicGenWrapper:
    def generate(self, prompt: str, duration: int = 30) -> str:
        """生成音乐"""
        output_path = f"data/outputs/musicgen_{hash(prompt) % 10000}.wav"
        return output_path
