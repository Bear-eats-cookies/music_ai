"""
人声分离模块 - 使用 Demucs
从混合音频中分离人声和伴奏
"""
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class VocalSeparatorDemucs:
    """基于Demucs的人声分离器"""

    def __init__(self, model_name: str = "htdemucs", device: str = None):
        """
        初始化人声分离器

        Args:
            model_name: Demucs模型名称 ('htdemucs', 'htdemucs_ft', 'htdemucs_6s')
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.project_root = Path(__file__).parent.parent.parent

        # Demucs模型
        self.separator = None

        print(f"[人声分离] 初始化Demucs分离器...")
        print(f"  模型: {model_name}")
        print(f"  设备: {self.device}")

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载Demucs模型"""
        try:
            from demucs import pretrained

            print(f"  正在加载Demucs模型...")
            local_repo = self.project_root / "models" / "demucs"
            repo = local_repo if local_repo.exists() else None
            if repo is not None:
                print(f"  使用本地Demucs权重目录: {repo}")
            try:
                self.separator = pretrained.get_model(self.model_name, repo=repo)
            except Exception as e:
                if repo is None:
                    raise
                print(f"  ⚠ 按模型名加载失败: {e}")
                # 部分本地导出的Demucs权重只有签名文件，没有bag配置。
                # 这种情况下回退到可加载的签名模型，优先使用已验证可分离四轨的权重。
                candidate_names = ["955717e8"]
                candidate_names.extend(
                    path.stem.split("-")[0]
                    for path in sorted(repo.glob("*.th"))
                )
                last_error = e
                for candidate in candidate_names:
                    try:
                        self.separator = pretrained.get_model(candidate, repo=repo)
                        print(f"  ✓ 已改用本地签名模型: {candidate}")
                        break
                    except Exception as inner_e:
                        last_error = inner_e
                else:
                    raise last_error
            self.separator = self.separator.to(self.device)
            self.separator.eval()
            print(f"  ✓ Demucs模型加载成功")

        except ImportError:
            print(f"  ⚠ Demucs未安装，将使用模拟模式")
        except Exception as e:
            print(f"  ⚠ Demucs加载失败: {e}")
            import traceback
            traceback.print_exc()

    def separate(
        self,
        audio_path: str,
        output_dir: str = None,
        sample_rate: int = 48000
    ) -> Dict[str, str]:
        """
        分离音频中的人声和伴奏

        Args:
            audio_path: 输入音频路径
            output_dir: 输出目录（可选）
            sample_rate: 输出采样率

        Returns:
            {
                "vocals": "人声路径",
                "drums": "鼓点路径",
                "bass": "贝斯路径",
                "other": "其他乐器路径",
                "accompaniment": "伴奏路径"  # drums + bass + other
            }
        """
        if output_dir is None:
            output_dir = str(Path(audio_path).parent)

        print(f"\n[人声分离] 开始分离...")
        print(f"  输入: {audio_path}")
        print(f"  输出目录: {output_dir}")

        # 尝试使用真实Demucs模型
        if self.separator is not None:
            try:
                result = self._separate_with_demucs(audio_path, output_dir, sample_rate)
                if result:
                    return result
            except Exception as e:
                print(f"  ⚠ Demucs分离失败: {e}")

        # 备用: 模拟分离
        return self._simulate_separation(audio_path, output_dir, sample_rate)

    def _separate_with_demucs(
        self,
        audio_path: str,
        output_dir: str,
        sample_rate: int
    ) -> Optional[Dict[str, str]]:
        """使用Demucs进行分离"""
        try:
            import torchaudio

            # 加载音频
            waveform, sr = torchaudio.load(audio_path)
            if sr != 44100 and sr != 48000:
                # 重采样到44100（Demucs的标准采样率）
                import torchaudio.transforms as T
                resampler = T.Resample(sr, 44100)
                waveform = resampler(waveform)
                sr = 44100

            # 添加batch维度
            waveform = waveform.unsqueeze(0).to(self.device)

            print(f"  正在分离...")
            with torch.no_grad():
                sources = self.separator(waveform)

            # 移除batch维度并转到CPU
            sources = sources.squeeze(0).cpu()

            # 保存各个音轨
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            stem_names = ["drums", "bass", "other", "vocals"]
            output_paths = {}

            for i, stem_name in enumerate(stem_names):
                stem_audio = sources[i].numpy().T
                output_path = output_dir / f"{Path(audio_path).stem}_{stem_name}.wav"

                # 重采样到目标采样率
                if sample_rate != sr:
                    import scipy.signal as signal
                    stem_audio = signal.resample(
                        stem_audio,
                        int(len(stem_audio) * sample_rate / sr)
                    )

                sf.write(output_path, stem_audio, sample_rate)
                output_paths[stem_name] = str(output_path)
                print(f"  ✓ {stem_name}: {output_path}")

            # 创建伴奏（drums + bass + other）
            accompaniment = output_paths["drums"]
            accompaniment_audio = sf.read(accompaniment, dtype='float32')[0]

            for stem in ["bass", "other"]:
                stem_audio, _ = sf.read(output_paths[stem], dtype='float32')
                min_len = min(len(accompaniment_audio), len(stem_audio))
                accompaniment_audio[:min_len] += stem_audio[:min_len]

            # 归一化
            peak = np.max(np.abs(accompaniment_audio))
            if peak > 0:
                accompaniment_audio = accompaniment_audio / peak * 0.95

            accompaniment_path = output_dir / f"{Path(audio_path).stem}_accompaniment.wav"
            sf.write(accompaniment_path, accompaniment_audio, sample_rate)
            output_paths["accompaniment"] = str(accompaniment_path)
            print(f"  ✓ accompaniment: {accompaniment_path}")

            return output_paths

        except Exception as e:
            print(f"  ⚠ Demucs分离失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _simulate_separation(
        self,
        audio_path: str,
        output_dir: str,
        sample_rate: int
    ) -> Dict[str, str]:
        """模拟分离（备用方案）"""
        print(f"  使用模拟分离模式...")

        # 读取原始音频
        audio, sr = sf.read(audio_path)

        # 如果需要重采样
        if sample_rate != sr:
            import scipy.signal as signal
            audio = signal.resample(audio, int(len(audio) * sample_rate / sr))
            sr = sample_rate

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建模拟的分离音轨
        # 在实际应用中，Demucs会使用深度学习来分离
        # 这里我们使用简单的频率过滤来模拟

        # 人声: 中频段
        vocals = self._simulate_vocals(audio, sr)
        vocals_path = output_dir / f"{Path(audio_path).stem}_vocals.wav"
        sf.write(vocals_path, vocals, sr)

        # 伴奏: 低频+高频
        accompaniment = self._simulate_accompaniment(audio, sr)
        accompaniment_path = output_dir / f"{Path(audio_path).stem}_accompaniment.wav"
        sf.write(accompaniment_path, accompaniment, sr)

        # 鼓点: 低频+瞬态
        drums = self._simulate_drums(audio, sr)
        drums_path = output_dir / f"{Path(audio_path).stem}_drums.wav"
        sf.write(drums_path, drums, sr)

        # 贝斯: 低频
        bass = self._simulate_bass(audio, sr)
        bass_path = output_dir / f"{Path(audio_path).stem}_bass.wav"
        sf.write(bass_path, bass, sr)

        # 其他乐器: 高频
        other = self._simulate_other(audio, sr)
        other_path = output_dir / f"{Path(audio_path).stem}_other.wav"
        sf.write(other_path, other, sr)

        print(f"  ✓ vocals: {vocals_path}")
        print(f"  ✓ accompaniment: {accompaniment_path}")
        print(f"  ✓ drums: {drums_path}")
        print(f"  ✓ bass: {bass_path}")
        print(f"  ✓ other: {other_path}")

        return {
            "vocals": str(vocals_path),
            "accompaniment": str(accompaniment_path),
            "drums": str(drums_path),
            "bass": str(bass_path),
            "other": str(other_path)
        }

    def _simulate_vocals(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """模拟人声分离"""
        import scipy.signal as signal

        # 使用带通滤波器保留中频（人声主要频率范围）
        nyquist = sr / 2
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        # 如果是立体声，分别处理每个声道
        if len(audio.shape) == 1:
            vocals = signal.filtfilt(b, a, audio)
        else:
            vocals = np.zeros_like(audio)
            for channel in range(audio.shape[1]):
                vocals[:, channel] = signal.filtfilt(b, a, audio[:, channel])

        # 归一化
        peak = np.max(np.abs(vocals))
        if peak > 0:
            vocals = vocals / peak * 0.9

        return vocals

    def _simulate_accompaniment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """模拟伴奏分离"""
        import scipy.signal as signal

        # 伴奏 = 原始音频 - 人声（简化处理）
        vocals = self._simulate_vocals(audio, sr)
        accompaniment = audio - vocals

        # 归一化
        peak = np.max(np.abs(accompaniment))
        if peak > 0:
            accompaniment = accompaniment / peak * 0.9

        return accompaniment

    def _simulate_drums(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """模拟鼓点分离"""
        import scipy.signal as signal

        # 鼓点主要是低频+瞬态
        nyquist = sr / 2
        low = 50 / nyquist
        high = 200 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        if len(audio.shape) == 1:
            drums = signal.filtfilt(b, a, audio)
        else:
            drums = np.zeros_like(audio)
            for channel in range(audio.shape[1]):
                drums[:, channel] = signal.filtfilt(b, a, audio[:, channel])

        # 归一化
        peak = np.max(np.abs(drums))
        if peak > 0:
            drums = drums / peak * 0.9

        return drums

    def _simulate_bass(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """模拟贝斯分离"""
        import scipy.signal as signal

        # 贝斯是低频
        nyquist = sr / 2
        low = 40 / nyquist
        high = 250 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        if len(audio.shape) == 1:
            bass = signal.filtfilt(b, a, audio)
        else:
            bass = np.zeros_like(audio)
            for channel in range(audio.shape[1]):
                bass[:, channel] = signal.filtfilt(b, a, audio[:, channel])

        # 归一化
        peak = np.max(np.abs(bass))
        if peak > 0:
            bass = bass / peak * 0.9

        return bass

    def _simulate_other(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """模拟其他乐器分离"""
        import scipy.signal as signal

        # 其他乐器主要是高频
        nyquist = sr / 2
        low = 1000 / nyquist
        b, a = signal.butter(4, low, btype='high')

        if len(audio.shape) == 1:
            other = signal.filtfilt(b, a, audio)
        else:
            other = np.zeros_like(audio)
            for channel in range(audio.shape[1]):
                other[:, channel] = signal.filtfilt(b, a, audio[:, channel])

        # 归一化
        peak = np.max(np.abs(other))
        if peak > 0:
            other = other / peak * 0.9

        return other

    def is_ready(self) -> bool:
        """检查分离器是否就绪"""
        return self.separator is not None
