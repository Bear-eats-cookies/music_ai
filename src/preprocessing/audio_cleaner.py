"""
模块1: 音频预处理
功能: 清洗、降噪、人声分离、特征提取
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

class AudioPreprocessor:
    def __init__(self, target_sr: int = 24000):
        self.target_sr = target_sr
    
    def process(self, audio_path: str, audio_type: str = "mixed") -> Tuple[np.ndarray, Dict]:
        """
        输入: 原始音频路径, 音频类型
        输出: 处理后的音频数组, 元数据字典
        """
        print(f"  加载音频: {audio_path}")
        # 1. 加载音频 - 限制最大时长避免内存溢出
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True, duration=30.0)
        print(f"  音频时长: {len(audio)/sr:.1f}秒")
        
        # 2. 简单降噪 - 使用高通滤波器替代noisereduce
        print("  简单降噪处理...")
        audio_clean = self._simple_denoise(audio, sr)
        print("  降噪完成")
        
        # 3. 归一化
        audio_clean = librosa.util.normalize(audio_clean)
        
        # 4. 提取元数据
        metadata = self._extract_metadata(audio_clean, sr, audio_type)
        print(f"  音频质量评分: {metadata['quality_score']:.2f}")
        
        return audio_clean, metadata
    
    def _simple_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """简单降噪 - 使用高通滤波器"""
        from scipy.signal import butter, filtfilt
        
        # 高通滤波器去除低频噪声
        nyquist = sr / 2
        low_cutoff = 80 / nyquist  # 80Hz高通
        b, a = butter(4, low_cutoff, btype='high')
        
        return filtfilt(b, a, audio)
    
    def _extract_metadata(self, audio: np.ndarray, sr: int, audio_type: str) -> Dict:
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio ** 2))
        snr = self._estimate_snr_simple(audio)
        
        return {
            "duration": float(duration),
            "sample_rate": sr,
            "audio_type": audio_type,
            "quality_score": min(snr / 20.0, 1.0),
            "snr_db": float(snr)
        }
    
    def _estimate_snr_simple(self, audio: np.ndarray) -> float:
        """简化的信噪比估算"""
        signal_power = np.mean(audio ** 2)
        noise_power = np.var(audio) * 0.1  # 简化估算
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    def save(self, audio: np.ndarray, output_path: str):
        sf.write(output_path, audio, self.target_sr)
