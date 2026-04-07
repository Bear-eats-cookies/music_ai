"""
SVC (Singing Voice Conversion) 转换器
使用 RVC 模型进行音色转换，实现演唱人声的音色替换
"""
import torch
import numpy as np
import soundfile as sf
import shutil
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from ..voice_cloning.rvc_runtime import (
    discover_user_models,
    inspect_rvc_runtime,
    inspect_user_model,
    select_rvc_model,
)


class SVCConverter:
    """SVC转换器 - 使用RVC进行音色转换"""

    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化SVC转换器

        Args:
            model_path: RVC模型路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.project_root = Path(__file__).parent.parent.parent
        self.model_path = model_path

        # RVC模型组件
        self.hubert = None
        self.rvc_model = None
        self.rmvpe = None
        self.rvc_inference = None
        self.runtime_info = inspect_rvc_runtime(self.project_root / "models" / "RVC1006Nvidia")
        self.available_user_models = []
        self.invalid_user_models = []
        self.last_error = None

        print(f"[SVC转换器] 初始化...")
        print(f"  设备: {self.device}")
        print(f"  项目根目录: {self.project_root}")

        # 尝试加载RVC模型
        self._load_rvc_model()

    def _load_rvc_model(self):
        """加载RVC模型"""
        discovery = discover_user_models(self.project_root / "models" / "user_voices")
        self.available_user_models = discovery["valid_models"]
        self.invalid_user_models = discovery["invalid_models"]
        self.last_error = None

        if not self.runtime_info["ready"]:
            missing = ", ".join(self.runtime_info["missing_files"])
            self.last_error = f"RVC运行时未就绪: 缺少 {missing}"
            print(f"  ⚠ {self.last_error}")
        elif self.model_path:
            print(f"  检查用户RVC模型: {self.model_path}")
            self._try_load_model(self.model_path)
        else:
            default_model = select_rvc_model(allow_runtime_default=True)
            if self.available_user_models:
                print(f"  找到 {len(self.available_user_models)} 个真实RVC用户模型")
                self._try_load_model(self.available_user_models[0]["model_path"])
            elif default_model is not None:
                print(f"  未找到用户模型，先尝试官方默认RVC音色: {Path(default_model['model_path']).name}")
                self._try_load_model(default_model["model_path"])
            else:
                self.last_error = "未找到真实RVC用户模型"
                print(f"  ⚠ {self.last_error}")

        if self.invalid_user_models:
            invalid_names = ", ".join(Path(info["model_path"]).name for info in self.invalid_user_models[:3])
            print(f"  提示: 这些 `.pth` 不是可用RVC用户模型: {invalid_names}")

    def _try_load_model(self, model_path: str):
        """尝试加载RVC模型"""
        model_info = inspect_user_model(model_path)
        if not model_info["valid"]:
            self.last_error = f"用户模型不可用: {model_info['reason']}"
            print(f"  ⚠ {self.last_error}")
            return

        try:
            from ..voice_cloning.rvc_inference import RVCInference

            self.rvc_inference = RVCInference()
            if self.rvc_inference.load_model(model_path):
                self.rvc_model = self.rvc_inference.current_model_info
                print(f"  ✓ RVC模型加载成功")
            else:
                self.last_error = self.rvc_inference.last_error or "RVC模型不可用于真实推理"
                print(f"  ⚠ {self.last_error}")
        except Exception as e:
            self.last_error = f"RVC模型加载失败: {e}"
            print(f"  ⚠ {self.last_error}")

    def convert(
        self,
        source_audio_path: str,
        target_voice_profile: Dict,
        output_path: str = None,
        f0_shift: int = 0,
        f0_method: str = "rmvpe"
    ) -> str:
        """
        转换音频音色

        Args:
            source_audio_path: 源音频路径（需要转换的演唱人声）
            target_voice_profile: 目标声音档案
            output_path: 输出路径（可选）
            f0_shift: 音高偏移（半音）
            f0_method: 音高提取方法 ('rmvpe', 'crepe', 'pm')

        Returns:
            转换后的音频路径
        """
        if output_path is None:
            output_path = str(Path(source_audio_path).parent / f"converted_{Path(source_audio_path).name}")

        print(f"\n[SVC转换] 开始音色转换...")
        print(f"  源音频: {source_audio_path}")
        print(f"  输出路径: {output_path}")
        print(f"  音高偏移: {f0_shift} 半音")
        print(f"  音高提取方法: {f0_method}")

        target_model_path = target_voice_profile.get("voice_model_path")
        if target_model_path:
            self._try_load_model(target_model_path)

        if self.rvc_model is not None and self.rvc_inference is not None:
            try:
                result = self._convert_with_rvc(
                    source_audio_path,
                    target_voice_profile,
                    output_path,
                    f0_shift,
                    f0_method
                )
                if result:
                    return result
            except Exception as e:
                print(f"  ⚠ RVC转换失败: {e}")

        if self.last_error:
            print(f"  ⚠ 跳过RVC转换: {self.last_error}")
        elif target_model_path:
            model_info = inspect_user_model(target_model_path)
            if not model_info["valid"]:
                print(f"  ⚠ 跳过RVC转换: {model_info['reason']}")

        # 备用: 保留原始人声，避免伪造音色转换结果
        return self._passthrough_conversion(source_audio_path, output_path)

    def _convert_with_rvc(
        self,
        source_audio_path: str,
        target_voice_profile: Dict,
        output_path: str,
        f0_shift: int,
        f0_method: str
    ) -> Optional[str]:
        """使用RVC模型进行转换"""
        try:
            # 获取目标模型路径
            target_model_path = target_voice_profile.get("voice_model_path")
            if target_model_path and Path(target_model_path).exists():
                print(f"  使用目标模型: {target_model_path}")
                if not self.rvc_inference.load_model(target_model_path):
                    return None

            # 执行转换
            converted_audio = self.rvc_inference.convert(
                source_audio_path,
                user_model_path=target_model_path,
                f0_shift=f0_shift,
                f0_method=f0_method
            )

            if converted_audio is not None:
                if isinstance(converted_audio, tuple):
                    converted_audio, sample_rate = converted_audio
                else:
                    sample_rate = 48000
                sf.write(output_path, converted_audio, sample_rate)
                print(f"  ✓ RVC转换成功: {output_path}")
                return output_path

        except Exception as e:
            print(f"  ⚠ RVC转换失败: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _passthrough_conversion(
        self,
        source_audio_path: str,
        output_path: str
    ) -> str:
        """保留原始人声，避免输出伪造的音色转换结果。"""
        print(f"  保留原始人声（RVC未就绪）...")
        shutil.copy(source_audio_path, output_path)
        print(f"  ✓ 已保留原始人声: {output_path}")
        return output_path

    def extract_features(self, audio_path: str) -> Dict:
        """
        提取音频特征

        Args:
            audio_path: 音频路径

        Returns:
            特征字典
        """
        try:
            import librosa

            audio, sr = librosa.load(audio_path, sr=48000)

            # 提取音高
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )

            # 提取MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

            # 提取色度特征
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

            # 提取频谱质心
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

            # 提取零交叉率
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

            return {
                "f0": f0,
                "f0_mean": float(np.nanmean(f0)) if len(f0) > 0 else 0.0,
                "f0_std": float(np.nanstd(f0)) if len(f0) > 0 else 0.0,
                "mfcc": mfcc,
                "chroma": chroma,
                "spectral_centroid": float(np.mean(spectral_centroid)),
                "zero_crossing_rate": float(np.mean(zero_crossing_rate)),
                "duration": float(len(audio) / sr),
                "sample_rate": sr
            }

        except Exception as e:
            print(f"  ⚠ 特征提取失败: {e}")
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "duration": 0.0,
                "sample_rate": 48000
            }

    def is_ready(self) -> bool:
        """检查转换器是否就绪"""
        return self.rvc_model is not None
