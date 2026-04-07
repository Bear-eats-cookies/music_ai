"""母带处理"""
import librosa
import numpy as np

class Mastering:
    def master(self, audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
        """母带处理"""
        audio = librosa.util.normalize(audio)
        audio = np.tanh(audio * 0.95) / 0.95
        return audio
