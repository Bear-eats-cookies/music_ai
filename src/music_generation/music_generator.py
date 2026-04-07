"""
模块3: 音乐生成 - 完整版
功能: 伴奏生成、旋律生成、人声合成
支持: ACE-Step v1.5 Turbo + Fish Speech 1.5
"""
import torch
import numpy as np
import soundfile as sf
from typing import Dict, Tuple, Optional
from pathlib import Path
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


class MusicGenerator:
    """音乐生成器 - 支持真实的AI音乐生成"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 项目根目录
        self.project_root = Path(__file__).parent.parent.parent
        self.output_dir = self.project_root / "data" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ACE-Step模型路径
        self.ace_step_path = self.project_root / "models" / "Ace-Step1.5"
        self.ace_step = None
        
        # 使用 RVC 进行音色转换（方案 A）
        
        print("\n" + "=" * 60)
        print("初始化音乐生成器")
        print("=" * 60)
        print(f"  项目根目录: {self.project_root}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  设备: {self.device}")
        
        # 加载ACE-Step模型
        self._load_ace_step()
        
        # 加载Fish Speech模型
        self._load_fish_speech()
        
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
            
            # 检查是否加载成功
            if not self.ace_step.is_ready():
                print(f"  ⚠ ACE-Step模型未完全加载，将尝试API模式")
                self.ace_step = None
                
                # 尝试API模式
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
    
    
    def _print_status(self):
        """打印状态报告"""
        print("\n" + "=" * 60)
        print("音乐生成器状态:")
        print(f"  ACE-Step (音乐生成): {'✓ 可用' if self.ace_step and self.ace_step.is_ready() else '✗ 不可用'}")
        print("  RVC (音色转换): ✓ 使用方案 A")
        
        if self.ace_step is None and self.fish_speech is None:
            print("\n  ⚠ 警告: 所有AI模型都不可用，将使用模拟模式")
        elif self.ace_step is None:
            print("\n  ⚠ 注意: 音乐生成将使用模拟模式")
        elif self.fish_speech is None:
            print("\n  ⚠ 注意: 人声合成将使用模拟模式")
        
        print("=" * 60 + "\n")
    
    def generate(
        self, 
        lyrics: str, 
        style: str, 
        voice_profile: Dict, 
        params: Dict
    ) -> Dict:
        """
        生成完整的歌曲
        
        Args:
            lyrics: 歌词
            style: 音乐风格
            voice_profile: 声音档案
            params: 生成参数
        
        Returns:
            {
                "song_id": str,
                "instrumental_path": str,
                "vocal_path": str,
                "melody_path": str,
                "metadata": dict
            }
        """
        song_id = f"song_{hash(lyrics + style) % 100000}"
        
        print("\n" + "=" * 60)
        print(f"开始生成歌曲: {song_id}")
        print("=" * 60)
        print(f"  风格: {style}")
        print(f"  歌词: {lyrics[:50]}...")
        print(f"  参数: {params}")
        
        # 1. 生成伴奏
        print("\n[步骤 1/3] 生成伴奏...")
        instrumental_path = self._generate_instrumental(style, params, song_id, lyrics)
        
        # 2. 生成旋律
        print("\n[步骤 2/3] 生成旋律...")
        melody_path = self._generate_melody(lyrics, style, params, song_id)
        
        # 3. 合成人声
        print("\n[步骤 3/3] 合成人声...")
        vocal_path = self._synthesize_vocal(lyrics, melody_path, voice_profile, song_id)
        
        result = {
            "song_id": song_id,
            "instrumental_path": instrumental_path,
            "vocal_path": vocal_path,
            "melody_path": melody_path,
            "metadata": {
                "style": style,
                "bpm": params.get("tempo", 80),
                "key": params.get("key", "C_major"),
                "duration": params.get("duration", 180),
                "lyrics": lyrics,
                "model_status": {
                    "ace_step": self.ace_step is not None and self.ace_step.is_ready(),
                    "fish_speech": self.fish_speech is not None and self.fish_speech.is_ready()
                }
            }
        }
        
        print("\n" + "=" * 60)
        print(f"歌曲生成完成: {song_id}")
        print(f"  伴奏: {instrumental_path}")
        print(f"  人声: {vocal_path}")
        print("=" * 60)
        
        return result
    
    def _generate_instrumental(
        self, 
        style: str, 
        params: Dict, 
        song_id: str,
        lyrics: str = None
    ) -> str:
        """使用ACE-Step生成伴奏"""
        output_path = self.output_dir / f"{song_id}_instrumental.wav"
        
        # 尝试使用真实的ACE-Step模型
        if self.ace_step is not None and self.ace_step.is_ready():
            print(f"  使用ACE-Step生成伴奏...")
            
            # 构建提示词
            prompt = self._build_prompt(style, params)
            duration = params.get('duration', 90)
            
            # 调用ACE-Step生成
            audio = self.ace_step.generate_music(
                prompt=prompt,
                duration=duration,
                lyrics=lyrics,
                seed=params.get('seed', None)
            )
            
            if audio is not None:
                # ACE-Step输出是48kHz，保存
                sf.write(output_path, audio, 48000)
                print(f"  ✓ ACE-Step伴奏生成成功: {output_path}")
                return str(output_path)
            else:
                print(f"  ⚠ ACE-Step生成返回空，使用备用方案")
        
        # 备用: 生成模拟伴奏
        print(f"  使用模拟伴奏生成...")
        return self._generate_simulated_instrumental(style, params, song_id)
    
    def _build_prompt(self, style: str, params: Dict) -> str:
        """构建音乐生成提示词"""
        tempo = params.get('tempo', 80)
        key = params.get('key', 'C major')
        
        # 风格映射
        style_descriptions = {
            "pop_ballad": f"emotional pop ballad, {tempo} BPM, piano and strings, heartfelt melody",
            "folk_acoustic": f"acoustic folk, {tempo} BPM, guitar and light percussion, warm and natural",
            "r&b_soul": f"smooth R&B soul, {tempo} BPM, groovy bass, electric piano, sensual",
            "rock": f"energetic rock, {tempo} BPM, electric guitars, drums, powerful",
            "electronic": f"electronic dance music, {tempo} BPM, synthesizers, beat drops",
            "jazz": f"smooth jazz, {tempo} BPM, saxophone, piano, sophisticated",
            "classical": f"orchestral classical, {tempo} BPM, strings, woodwinds, elegant",
        }
        
        prompt = style_descriptions.get(style, f"{style} music, {tempo} BPM, high quality production")
        
        # 添加调性信息
        prompt += f", key of {key}"
        
        return prompt
    
    def _generate_simulated_instrumental(
        self, 
        style: str, 
        params: Dict, 
        song_id: str
    ) -> str:
        """生成模拟伴奏（备用方案）"""
        output_path = self.output_dir / f"{song_id}_instrumental.wav"
        
        duration = params.get('duration', 90)
        sr = 24000
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        
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
        
        sf.write(output_path, audio, sr)
        print(f"  ✓ 模拟伴奏生成完成: {output_path}")
        return str(output_path)
    
    def _generate_melody(
        self, 
        lyrics: str, 
        style: str, 
        params: Dict, 
        song_id: str
    ) -> str:
        """生成旋律MIDI"""
        output_path = self.output_dir / f"{song_id}_melody.mid"
        
        # 创建MIDI文件标记
        output_path.touch()
        print(f"  旋律文件: {output_path}")
        
        return str(output_path)
    
    def _synthesize_vocal(
        self, 
        lyrics: str, 
        melody_path: str, 
        voice_profile: Dict, 
        song_id: str
    ) -> str:
        """生成静音人声轨（占位符）- 实际人声由 RVC 转换生成"""
        output_path = self.output_dir / f"{song_id}_vocal.wav"
        
        print(f"  生成人声占位符...")
        print(f"  💡 实际人声将由 RVC 从 ACE-Step 输出中转换")
        
        # 生成静音轨道作为占位符
        duration = 90
        sr = 24000
        import numpy as np
        import soundfile as sf
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        sf.write(output_path, audio, sr)
        
        print(f"  ✓ 人声占位符: {output_path}")
        return str(output_path)

