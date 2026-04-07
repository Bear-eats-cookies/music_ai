"""音频特征提取"""
import librosa
import numpy as np

class FeatureExtractor:
    def extract(self, audio: np.ndarray, sr: int) -> dict:
        """提取音频特征"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        
        return {
            "mfcc": mfcc.mean(axis=1).tolist(),
            "spectral_centroid": float(np.mean(spectral_centroid))
        }
