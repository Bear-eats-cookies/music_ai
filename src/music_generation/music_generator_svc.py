"""
模块3: 音乐生成 - 重构版（方案A）
功能: ACE-Step生成完整演唱歌曲 + SVC音色转换
架构: ACE-Step -> 人声分离 -> SVC转换 -> 重新混音
"""
import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from .ace_step_wrapper import ACEStepWrapper
    ACE_LOCAL_AVAILABLE = True
except:
    ACE_LOCAL_AVAILABLE = False

try:
    from .ace_step_gradio import ACEStepGradioWrapper
    ACE_API_AVAILABLE = True
except:
    ACE_API_AVAILABLE = False


class MusicGeneratorSVC:
    """音乐生成器 - 方案A架构"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.allow_fake_audio = os.getenv("MUSIC_AI_ALLOW_FAKE_AUDIO", "0") == "1"

        # 项目根目录
        self.project_root = Path(__file__).parent.parent.parent
        self.output_dir = self.project_root / "data" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ACE-Step模型
        self.ace_step = None
        self.ace_step_path = self.project_root / "models" / "Ace-Step1.5"

        # 人声分离器
        self.vocal_separator = None

        # SVC转换器
        self.svc_converter = None

        print("\n" + "=" * 60)
        print("初始化音乐生成器（方案A架构）")
        print("=" * 60)
        print(f"  项目根目录: {self.project_root}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  设备: {self.device}")

        # 加载各个组件
        self._load_ace_step()
        self._load_vocal_separator()
        self._load_svc_converter()

        # 状态报告
        self._print_status()

    def _load_ace_step(self):
        """加载ACE-Step音乐生成模型"""
        print(f"\n[音乐生成] ACE-Step模型路径: {self.ace_step_path}")

        if not self.ace_step_path.exists():
            print(f"  ⚠ ACE-Step模型目录不存在")
            return

        if not ACE_LOCAL_AVAILABLE:
            print(f"  ⚠ ACE-Step包装器不可用")
            return

        try:
            print(f"  正在加载ACE-Step模型...")
            self.ace_step = ACEStepWrapper(str(self.ace_step_path))

            if not self.ace_step.is_ready():
                print(f"  ⚠ ACE-Step模型未完全加载，将尝试API模式")
                self.ace_step = None

                if ACE_API_AVAILABLE:
                    try:
                        self.ace_step = ACEStepGradioWrapper()
                        print(f"  ✓ ACE-Step API模式连接成功")
                    except Exception as e:
                        print(f"  ⚠ API连接失败: {e}")
            else:
                print(f"  ✓ ACE-Step本地模型加载成功")

        except Exception as e:
            print(f"  ⚠ ACE-Step加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.ace_step = None

    def _load_vocal_separator(self):
        """加载人声分离器"""
        print(f"\n[人声分离] 初始化Demucs分离器...")

        try:
            from ..preprocessing.vocal_separator_demucs import VocalSeparatorDemucs
            self.vocal_separator = VocalSeparatorDemucs(device=self.device)

            if self.vocal_separator.is_ready():
                print(f"  ✓ Demucs分离器加载成功")
            else:
                print(f"  ⚠ Demucs分离器将使用模拟模式")

        except Exception as e:
            print(f"  ⚠ 人声分离器加载失败: {e}")
            import traceback
            traceback.print_exc()

    def _load_svc_converter(self):
        """加载SVC转换器"""
        print(f"\n[SVC转换] 初始化RVC转换器...")

        try:
            from ..voice_conversion.svc_converter import SVCConverter
            self.svc_converter = SVCConverter(device=self.device)

            if self.svc_converter.is_ready():
                print(f"  ✓ RVC转换器加载成功")
            else:
                print(f"  ⚠ RVC转换器未就绪，生成时将保留原始人声")

        except Exception as e:
            print(f"  ⚠ SVC转换器加载失败: {e}")
            import traceback
            traceback.print_exc()

    def _print_status(self):
        """打印状态报告"""
        print("\n" + "=" * 60)
        print("音乐生成器状态（方案A架构）:")
        print(f"  ACE-Step (音乐生成): {'✓ 可用' if self.ace_step and self.ace_step.is_ready() else '✗ 不可用'}")
        if self.ace_step and getattr(self.ace_step, "official_backend", None) is not None:
            print(f"  ACE-Step后端: {getattr(self.ace_step, 'official_backend_name', 'official')}")
        print(f"  Demucs (人声分离): {'✓ 可用' if self.vocal_separator and self.vocal_separator.is_ready() else '✗ 不可用'}")
        print(f"  RVC (SVC转换): {'✓ 可用' if self.svc_converter and self.svc_converter.is_ready() else '✗ 不可用'}")

        if self.ace_step is None:
            print("\n  ⚠ 警告: ACE-Step不可用，音乐生成将使用模拟模式")
        if self.vocal_separator is None or not self.vocal_separator.is_ready():
            print("\n  ⚠ 注意: 人声分离将使用模拟模式")
        if self.svc_converter is None or not self.svc_converter.is_ready():
            print("\n  ⚠ 注意: SVC未就绪，将保留原始人声")

        print("=" * 60 + "\n")

    def generate(
        self,
        lyrics: str,
        style: str,
        voice_profile: Dict,
        params: Dict
    ) -> Dict:
        """
        生成完整的歌曲（方案A架构）

        流程:
        1. ACE-Step生成完整演唱歌曲（含人声）
        2. 使用Demucs分离人声和伴奏
        3. 使用RVC/SVC转换人声音色
        4. 重新混合转换后的人声和伴奏

        Args:
            lyrics: 歌词
            style: 音乐风格
            voice_profile: 声音档案
            params: 生成参数

        Returns:
            {
                "song_id": str,
                "original_song_path": str,  # 原始ACE-Step输出
                "vocal_path": str,          # 转换后的人声
                "instrumental_path": str,   # 分离的伴奏
                "final_song_path": str,     # 最终混合歌曲
                "metadata": dict
            }
        """
        song_id = f"song_{hash(lyrics + style) % 100000}"

        print("\n" + "=" * 60)
        print(f"开始生成歌曲（方案A架构）: {song_id}")
        print("=" * 60)
        print(f"  风格: {style}")
        print(f"  歌词: {lyrics[:50]}...")
        print(f"  参数: {params}")

        # 1. 使用ACE-Step生成完整演唱歌曲
        print("\n[步骤 1/4] 生成完整演唱歌曲...")
        original_song_path = self._generate_full_song(style, params, song_id, lyrics)

        # 2. 分离人声和伴奏
        print("\n[步骤 2/4] 分离人声和伴奏...")
        separated_paths = self._separate_vocals(original_song_path, song_id)

        # 3. 转换人声音色
        print("\n[步骤 3/4] 转换人声音色...")
        converted_vocal_path = self._convert_vocal(
            separated_paths["vocals"],
            voice_profile,
            song_id
        )

        # 4. 重新混合
        print("\n[步骤 4/4] 重新混合...")
        final_song_path = self._mix_final_song(
            converted_vocal_path,
            separated_paths["accompaniment"],
            song_id
        )

        result = {
            "song_id": song_id,
            "original_song_path": original_song_path,
            "vocal_path": converted_vocal_path,
            "instrumental_path": separated_paths["accompaniment"],
            "final_song_path": final_song_path,
            "metadata": {
                "style": style,
                "bpm": params.get("tempo", 80),
                "key": params.get("key", "C_major"),
                "duration": params.get("duration", 180),
                "lyrics": lyrics,
                "architecture": "方案A: ACE-Step + SVC",
                "model_status": {
                    "ace_step": self.ace_step is not None and self.ace_step.is_ready(),
                    "demucs": self.vocal_separator is not None and self.vocal_separator.is_ready(),
                    "rvc": self.svc_converter is not None and self.svc_converter.is_ready()
                }
            }
        }

        print("\n" + "=" * 60)
        print(f"歌曲生成完成: {song_id}")
        print(f"  原始歌曲: {original_song_path}")
        print(f"  转换人声: {converted_vocal_path}")
        print(f"  伴奏: {separated_paths['accompaniment']}")
        print(f"  最终歌曲: {final_song_path}")
        print("=" * 60)

        return result

    def _generate_full_song(
        self,
        style: str,
        params: Dict,
        song_id: str,
        lyrics: str = None
    ) -> str:
        """使用ACE-Step生成完整演唱歌曲"""
        output_path = self.output_dir / f"{song_id}_original.wav"

        # 尝试使用真实的ACE-Step模型
        if self.ace_step is not None and self.ace_step.is_ready():
            print(f"  使用ACE-Step生成完整演唱歌曲...")

            # 构建提示词
            prompt = self._build_prompt(style, params)
            duration = params.get('duration', 90)

            # 调用ACE-Step生成
            # 注意：这里ACE-Step会生成包含人声的完整歌曲
            audio = self.ace_step.generate_music(
                prompt=prompt,
                duration=duration,
                lyrics=lyrics,
                seed=params.get('seed', None)
            )

            if audio is not None:
                # ACE-Step输出是48kHz，保存
                sf.write(output_path, audio, 48000)
                print(f"  ✓ ACE-Step完整歌曲生成成功: {output_path}")
                return str(output_path)
            else:
                print(f"  ⚠ ACE-Step生成返回空")

        if self.allow_fake_audio:
            print(f"  使用模拟完整歌曲生成...")
            return self._generate_simulated_full_song(style, params, song_id)

        raise RuntimeError("ACE-Step 未就绪或生成失败，已禁止回退到模拟歌曲。")

    def _build_prompt(self, style: str, params: Dict) -> str:
        """构建音乐生成提示词"""
        tempo = params.get('tempo', 80)
        key = params.get('key', 'C major')

        # 风格映射 - 强调演唱
        style_descriptions = {
            "pop_ballad": f"emotional Chinese pop ballad, {tempo} BPM, piano and strings, full song, expressive lead vocal, clear verse and chorus",
            "folk_acoustic": f"acoustic folk ballad, {tempo} BPM, guitar and light percussion, full song, warm lead vocal, lyrical singing",
            "r&b_soul": f"smooth R&B soul, {tempo} BPM, groovy bass, electric piano, full song, soulful lead vocal singing",
            "rock": f"energetic rock anthem, {tempo} BPM, electric guitars and drums, full song, powerful lead vocal singing",
            "electronic": f"electronic pop song, {tempo} BPM, synthesizers and beat drops, full song, catchy lead vocal singing",
            "jazz": f"smooth jazz vocal song, {tempo} BPM, saxophone and piano, full song, intimate lead vocal singing",
            "classical": f"cinematic orchestral vocal piece, {tempo} BPM, strings and woodwinds, full song, dramatic sung melody",
        }

        prompt = style_descriptions.get(
            style,
            f"{style} full song, {tempo} BPM, high quality production, lead vocal singing the provided lyrics"
        )
        prompt += f", key of {key}, polished studio production"

        return prompt

    def _generate_simulated_full_song(
        self,
        style: str,
        params: Dict,
        song_id: str
    ) -> str:
        """生成模拟完整歌曲（备用方案）"""
        output_path = self.output_dir / f"{song_id}_original.wav"

        duration = params.get('duration', 90)
        sr = 48000  # 统一使用48kHz
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)

        # 生成伴奏
        instrumental = self._generate_simulated_instrumental_audio(style, params, t, sr)

        # 生成模拟人声
        vocal = self._generate_simulated_vocal_audio(style, params, t, sr)

        # 混合
        full_song = instrumental * 0.7 + vocal * 0.3

        # 归一化
        peak = np.max(np.abs(full_song))
        if peak > 0:
            full_song = full_song / peak * 0.95

        sf.write(output_path, full_song, sr)
        print(f"  ✓ 模拟完整歌曲生成完成: {output_path}")

        return str(output_path)

    def _generate_simulated_instrumental_audio(
        self,
        style: str,
        params: Dict,
        t: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """生成模拟伴奏音频"""
        duration = t[-1]

        def _softclip(x: np.ndarray, drive: float = 1.2) -> np.ndarray:
            return np.tanh(x * drive) / np.tanh(drive)

        def _triangle(phase: np.ndarray) -> np.ndarray:
            return 4.0 * np.abs(phase - 0.5) - 1.0

        # 根据风格选择和弦进行
        chord_progressions = {
            "pop_ballad": [
                [261.63, 329.63, 392.00],  # C
                [293.66, 369.99, 440.00],  # Dm
                [349.23, 440.00, 523.25],  # F
                [392.00, 493.88, 587.33],  # G
            ],
            "folk_acoustic": [
                [261.63, 329.63, 392.00],  # C
                [293.66, 369.99, 440.00],  # D
                [329.63, 415.30, 493.88],  # E
                [261.63, 329.63, 392.00],  # C
            ],
            "r&b_soul": [
                [220.00, 277.18, 329.63],  # Am
                [261.63, 329.63, 392.00],  # C
                [293.66, 369.99, 440.00],  # Dm
                [196.00, 246.94, 293.66],  # G
            ],
            "rock": [
                [261.63, 329.63, 392.00],  # C
                [261.63, 329.63, 392.00],  # C
                [349.23, 440.00, 523.25],  # F
                [392.00, 493.88, 587.33],  # G
            ],
        }

        chord_freqs = chord_progressions.get(style, chord_progressions["pop_ballad"])

        audio = np.zeros_like(t)
        chord_duration = duration / len(chord_freqs)
        bpm = float(params.get("tempo", 80))
        beat_hz = bpm / 60.0

        for i, freqs in enumerate(chord_freqs):
            start_idx = int(i * chord_duration * sr)
            end_idx = int((i + 1) * chord_duration * sr)
            if end_idx > len(t):
                end_idx = len(t)

            segment = t[start_idx:end_idx]
            seg_len = len(segment)
            seg_audio = np.zeros(seg_len, dtype=np.float32)

            # 低音
            bass_freq = freqs[0] / 2.0
            bass = np.sin(2 * np.pi * bass_freq * segment) * 0.18
            bass += np.sin(2 * np.pi * bass_freq * 2 * segment) * 0.06

            # Pad
            pad = np.zeros_like(segment, dtype=np.float32)
            for j, freq in enumerate(freqs):
                phase = (freq * segment) % 1.0
                pad += _triangle(phase).astype(np.float32) * (0.08 / (j + 1))
                pad += np.sin(2 * np.pi * freq * segment).astype(np.float32) * (0.04 / (j + 1))

            # 琶音
            arpeggio = np.zeros_like(segment, dtype=np.float32)
            if seg_len > 0:
                beat_samples = max(int(sr / beat_hz), 1)
                note_len = min(int(0.18 * sr), beat_samples)
                env = np.exp(-np.linspace(0, 5.0, note_len)).astype(np.float32)
                for k in range(0, seg_len, beat_samples):
                    note_freq = freqs[(k // beat_samples) % len(freqs)]
                    end_k = min(k + note_len, seg_len)
                    seg_t = segment[k:end_k]
                    arpeggio[k:end_k] += (np.sin(2 * np.pi * note_freq * seg_t) * env[: end_k - k] * 0.10).astype(np.float32)

            seg_audio += bass + pad + arpeggio

            # 侧链
            sidechain = 0.85 + 0.15 * np.sin(2 * np.pi * beat_hz * segment - np.pi / 2)
            seg_audio *= sidechain.astype(np.float32)

            seg_audio = _softclip(seg_audio, drive=1.4).astype(np.float32)
            audio[start_idx:end_idx] += seg_audio

        # 鼓点
        beat_samples = max(int(sr / beat_hz), 1)
        kick_len = int(0.12 * sr)
        snare_len = int(0.10 * sr)
        hat_len = int(0.03 * sr)
        noise = np.random.RandomState(0).randn(n).astype(np.float32)
        drums = np.zeros(n, dtype=np.float32)

        for k in range(0, n, beat_samples):
            beat_idx = (k // beat_samples) % 4
            if beat_idx in (0, 2):
                end = min(k + kick_len, n)
                tt = np.linspace(0, (end - k) / sr, end - k, endpoint=False)
                env = np.exp(-tt * 25.0).astype(np.float32)
                f0 = 90.0 - 40.0 * (tt / max(tt.max(), 1e-6))
                drums[k:end] += (np.sin(2 * np.pi * f0 * tt) * env * 0.25).astype(np.float32)
            if beat_idx in (1, 3):
                end = min(k + snare_len, n)
                tt = np.linspace(0, (end - k) / sr, end - k, endpoint=False)
                env = np.exp(-tt * 35.0).astype(np.float32)
                drums[k:end] += (noise[k:end] * env * 0.12).astype(np.float32)
            if (k // (beat_samples // 2 if beat_samples >= 2 else 1)) % 1 == 0:
                end = min(k + hat_len, n)
                tt = np.linspace(0, (end - k) / sr, end - k, endpoint=False)
                env = np.exp(-tt * 90.0).astype(np.float32)
                hn = noise[k:end]
                hn = np.concatenate([hn[:1], np.diff(hn)]).astype(np.float32)
                drums[k:end] += (hn * env * 0.05).astype(np.float32)

        audio = audio.astype(np.float32) + drums

        # 渐强渐弱
        fade_in = np.linspace(0, 1, int(2 * sr))
        fade_out = np.linspace(1, 0, int(3 * sr))
        audio[:len(fade_in)] *= fade_in
        audio[-len(fade_out):] *= fade_out

        # 归一化
        peak = float(np.max(np.abs(audio)) + 1e-9)
        audio = (audio / peak * 0.9).astype(np.float32)

        return audio

    def _generate_simulated_vocal_audio(
        self,
        style: str,
        params: Dict,
        t: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """生成模拟人声音频"""
        duration = t[-1]
        n = len(t)

        # 生成简单的正弦波模拟人声
        vocal = np.zeros(n, dtype=np.float32)

        # 根据风格选择基频范围
        base_freqs = {
            "pop_ballad": 220.0,  # A3
            "folk_acoustic": 196.0,  # G3
            "r&b_soul": 246.94,  # B3
            "rock": 261.63,  # C4
        }

        base_freq = base_freqs.get(style, 220.0)

        # 生成简单的旋律
        melody_pattern = [0, 2, 4, 5, 4, 2, 0, -2]  # 音程模式
        note_duration = duration / len(melody_pattern)

        for i, interval in enumerate(melody_pattern):
            start_idx = int(i * note_duration * sr)
            end_idx = int((i + 1) * note_duration * sr)
            if end_idx > n:
                end_idx = n

            freq = base_freq * (2 ** (interval / 12.0))
            segment = t[start_idx:end_idx]

            # 生成人声波形（带谐波）
            vocal_part = np.zeros_like(segment, dtype=np.float32)
            for harmonic in [1, 2, 3, 4]:
                amplitude = 1.0 / harmonic
                vocal_part += np.sin(2 * np.pi * freq * harmonic * segment).astype(np.float32) * amplitude

            # 添加颤音
            vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * segment)
            vocal_part *= vibrato.astype(np.float32)

            # 包络
            envelope = np.ones_like(segment)
            if len(envelope) > 0:
                envelope[:int(0.01 * sr)] = np.linspace(0, 1, min(int(0.01 * sr), len(envelope)))
                envelope[-int(0.01 * sr):] = np.linspace(1, 0, min(int(0.01 * sr), len(envelope)))
            vocal_part *= envelope.astype(np.float32)

            vocal[start_idx:end_idx] += vocal_part

        # 归一化
        peak = np.max(np.abs(vocal))
        if peak > 0:
            vocal = vocal / peak * 0.8

        return vocal

    def _separate_vocals(self, audio_path: str, song_id: str) -> Dict[str, str]:
        """分离人声和伴奏"""
        output_dir = self.output_dir / f"{song_id}_separated"

        if self.vocal_separator is not None and self.vocal_separator.is_ready():
            separated_paths = self.vocal_separator.separate(
                audio_path,
                output_dir=str(output_dir),
                sample_rate=48000
            )
            return separated_paths

        if self.allow_fake_audio:
            # 模拟分离
            print(f"  使用模拟分离...")
            audio, sr = sf.read(audio_path)

            # 创建简单的模拟分离
            output_dir.mkdir(parents=True, exist_ok=True)

            vocal_path = output_dir / f"{song_id}_vocals.wav"
            accompaniment_path = output_dir / f"{song_id}_accompaniment.wav"

            # 模拟人声（中频）
            import scipy.signal as signal
            nyquist = sr / 2
            low = 300 / nyquist
            high = 3400 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')

            if len(audio.shape) == 1:
                vocals = signal.filtfilt(b, a, audio)
            else:
                vocals = np.zeros_like(audio)
                for channel in range(audio.shape[1]):
                    vocals[:, channel] = signal.filtfilt(b, a, audio[:, channel])

            peak = np.max(np.abs(vocals))
            if peak > 0:
                vocals = vocals / peak * 0.9

            sf.write(vocal_path, vocals, sr)

            # 模拟伴奏
            accompaniment = audio - vocals
            peak = np.max(np.abs(accompaniment))
            if peak > 0:
                accompaniment = accompaniment / peak * 0.9

            sf.write(accompaniment_path, accompaniment, sr)

            return {
                "vocals": str(vocal_path),
                "accompaniment": str(accompaniment_path)
            }

        raise RuntimeError("Demucs 未就绪，已禁止回退到模拟人声分离。")

    def _convert_vocal(
        self,
        vocal_path: str,
        voice_profile: Dict,
        song_id: str
    ) -> str:
        """转换人声音色"""
        output_path = self.output_dir / f"{song_id}_converted_vocal.wav"

        if self.svc_converter is not None:
            converted_path = self.svc_converter.convert(
                vocal_path,
                voice_profile,
                output_path=str(output_path),
                f0_shift=0
            )
            return converted_path

        # 不伪造音色转换，直接保留原始人声
        print("  跳过RVC转换，保留原始人声...")
        import shutil
        shutil.copy(vocal_path, output_path)
        return str(output_path)

    def _mix_final_song(
        self,
        vocal_path: str,
        accompaniment_path: str,
        song_id: str
    ) -> str:
        """混合最终歌曲"""
        output_path = self.output_dir / f"{song_id}_final.wav"

        # 读取音轨
        vocal, sr1 = sf.read(vocal_path)
        accompaniment, sr2 = sf.read(accompaniment_path)

        # 确保采样率一致
        if sr1 != sr2:
            import scipy.signal as signal
            if sr1 != 48000:
                vocal = signal.resample(vocal, int(len(vocal) * 48000 / sr1))
                sr1 = 48000
            if sr2 != 48000:
                accompaniment = signal.resample(accompaniment, int(len(accompaniment) * 48000 / sr2))
                sr2 = 48000

        # 确保长度一致
        min_len = min(len(vocal), len(accompaniment))

        if len(vocal.shape) == 1:
            vocal = vocal.reshape(-1, 1)
        if len(accompaniment.shape) == 1:
            accompaniment = accompaniment.reshape(-1, 1)

        # 混合（人声音量稍大）
        vocal = vocal[:min_len] * 0.4
        accompaniment = accompaniment[:min_len] * 0.6

        mixed = vocal + accompaniment

        # 归一化
        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.95

        # 渐强渐弱
        fade_in = np.linspace(0, 1, int(2 * sr1))
        fade_out = np.linspace(1, 0, int(3 * sr1))
        mixed[:len(fade_in)] *= fade_in.reshape(-1, 1)
        mixed[-len(fade_out):] *= fade_out.reshape(-1, 1)

        sf.write(output_path, mixed, sr1)
        print(f"  ✓ 最终歌曲混合完成: {output_path}")

        return str(output_path)
