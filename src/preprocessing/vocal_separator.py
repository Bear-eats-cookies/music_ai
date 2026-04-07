"""人声分离"""
import librosa
import numpy as np

class VocalSeparator:
    def separate(self, audio_path: str) -> tuple:
        """分离人声和伴奏"""
        y, sr = librosa.load(audio_path, sr=22050)
        S = librosa.stft(y)
        S_harmonic, S_percussive = librosa.decompose.hpss(S)
        vocal = librosa.istft(S_harmonic)
        return vocal, sr
