"""特征分析器"""
import librosa
import numpy as np

class FeatureAnalyzer:
    def analyze(self, audio_path: str) -> dict:
        """分析音频特征"""
        y, sr = librosa.load(audio_path, sr=22050)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
        f0_valid = f0[~np.isnan(f0)]
        
        return {
            "f0_mean": float(np.mean(f0_valid)),
            "f0_std": float(np.std(f0_valid)),
            "f0_range": [float(f0_valid.min()), float(f0_valid.max())]
        }
