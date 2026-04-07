"""
Voice profile management for the music pipeline.

This module no longer creates fake RVC checkpoints or fake Fish Speech
embeddings.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch

from .rvc_inference import RVCInference, RVCTrainer
from .rvc_runtime import (
    discover_runtime_models,
    discover_user_models,
    select_rvc_model,
)

try:
    from .fish_speech_wrapper import FishSpeechWrapper
    FISH_AVAILABLE = True
except Exception:
    FISH_AVAILABLE = False


class VoiceCloner:
    def __init__(self, mode: str = "fish_speech"):
        """mode: `rvc` or `fish_speech`."""
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.project_root = Path(__file__).resolve().parents[2]

        self.fish_speech_path = self.project_root / "models" / "fish-speech-1.5"
        self.fish_speech = None

        print(f"  Fish Speech模型路径: {self.fish_speech_path}")
        if self.fish_speech_path.exists():
            print("  ✓ Fish Speech模型已找到")
            model_pth = self.fish_speech_path / "model.pth"
            if model_pth.exists():
                print(f"  ✓ 模型文件: {model_pth.stat().st_size / 1024 / 1024:.1f} MB")
                if FISH_AVAILABLE:
                    try:
                        self.fish_speech = FishSpeechWrapper(str(self.fish_speech_path))
                    except Exception as exc:
                        print(f"  ⚠ Fish Speech初始化失败: {exc}")
        else:
            print("  ⚠ Fish Speech模型未找到")

        self.rvc_trainer = None
        self.rvc_inference = None
        if mode == "rvc":
            self.rvc_trainer = RVCTrainer()
            self.rvc_inference = RVCInference()

    def train_or_encode(
        self,
        audio_path: str,
        user_id: str,
        training_mode: str = "quick",
        preferred_model_name: str | None = None,
        preferred_model_path: str | None = None,
    ) -> Dict:
        """
        Build or locate a voice profile.

        For RVC mode, only existing real user models are accepted.
        """
        if self.mode == "rvc":
            return self._resolve_existing_rvc_profile(
                user_id,
                preferred_model_name=preferred_model_name,
                preferred_model_path=preferred_model_path,
            )
        return self._encode_fish_speech(audio_path, user_id)

    def _resolve_existing_rvc_profile(
        self,
        user_id: str,
        preferred_model_name: str | None = None,
        preferred_model_path: str | None = None,
    ) -> Dict:
        """Return an existing real RVC voice profile if present."""
        allow_runtime_default = os.getenv("MUSIC_AI_USE_RUNTIME_DEFAULT_MODEL", "1") == "1"
        model_info = select_rvc_model(
            user_id=user_id,
            preferred_model_name=preferred_model_name,
            preferred_model_path=preferred_model_path,
            allow_runtime_default=allow_runtime_default,
        )
        if model_info is not None:
            print(f"  使用现有真实RVC用户模型: {model_info['model_path']}")
            return {
                "user_id": user_id,
                "voice_model_path": model_info["model_path"],
                "index_path": model_info["index_path"],
                "mode": "rvc_existing_model" if model_info.get("source") != "runtime_weights" else "rvc_runtime_default",
                "training_mode": "external",
                "quality_metrics": {},
                "source": model_info.get("source", "unknown"),
            }

        raise RuntimeError(self._build_missing_rvc_message())

    def _build_missing_rvc_message(self) -> str:
        """Build a concrete error describing why RVC cloning is unavailable."""
        reasons = []

        if self.rvc_inference is not None and not self.rvc_inference.is_runtime_ready():
            missing = ", ".join(self.rvc_inference.runtime_info["missing_files"])
            reasons.append(f"`models/RVC1006Nvidia` 缺少 {missing}")

        discovery = discover_user_models()
        if not discovery["valid_models"]:
            reasons.append("未找到真实RVC用户模型（需要手动放入训练好的 `.pth`，可选 `.index`）")

        runtime_models = discover_runtime_models()
        if runtime_models["valid_models"]:
            names = ", ".join(Path(info["model_path"]).stem for info in runtime_models["valid_models"][:4])
            reasons.append(f"官方RVC示例音色可用: {names}（不是你的声音）")

        invalid_names = [Path(info["model_path"]).name for info in discovery["invalid_models"]]
        if invalid_names:
            names = ", ".join(invalid_names[:3])
            reasons.append(f"`models/user_voices` 现有 `.pth` 不是可用的RVC用户模型: {names}")

        return "RVC未就绪：" + "；".join(reasons)

    def _encode_fish_speech(self, audio_path: str, user_id: str) -> Dict:
        """Encode a prompt with a real Fish Speech model if available."""
        if self.fish_speech is None:
            raise RuntimeError("Fish Speech未配置完成，已停止生成伪造的声音嵌入。")

        print("  使用Fish Speech进行声音编码...")
        prompt_path = self.project_root / "models" / "user_voices" / f"user_{user_id}_prompt.pt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            embedding = self.fish_speech.encode_prompt(audio_path)
            torch.save(embedding, prompt_path)
            print(f"  ✓ 声音特征已保存: {prompt_path}")
            return {
                "user_id": user_id,
                "prompt_embedding_path": str(prompt_path),
                "mode": "zero_shot",
                "model_path": str(self.fish_speech_path / "model.pth"),
            }
        except Exception as exc:
            raise RuntimeError(f"Fish Speech编码失败: {exc}") from exc

    def convert_voice(self, audio_path: str, voice_profile: Dict, f0_shift: int = 0) -> str:
        """Convert voice when a real backend is available."""
        if self.mode != "rvc" or self.rvc_inference is None:
            return audio_path

        converted_audio = self.rvc_inference.convert(
            audio_path,
            voice_profile.get("voice_model_path"),
            f0_shift,
        )
        return audio_path if converted_audio is None else converted_audio
