"""
RVC inference/training diagnostics.

This module refuses to fabricate fake user models or fake conversion results.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .rvc_runtime import inspect_rvc_runtime, inspect_user_model


class RVCInference:
    """RVC inference entrypoint placeholder with strict validation."""

    def __init__(self, model_dir: str = "models/RVC1006Nvidia"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = self.project_root / model_dir

        self.runtime_info = inspect_rvc_runtime(self.model_dir)
        self.current_model_info: Optional[Dict] = None
        self.last_error: Optional[str] = None
        self.runtime_root = Path(self.runtime_info["repo_path"])
        self.vc = None
        self.config = None

        print(f"  RVC模型目录: {self.model_dir}")
        if not self.runtime_info["ready"]:
            missing = ", ".join(self.runtime_info["missing_files"])
            print(f"  ⚠ RVC运行时未就绪: 缺少 {missing}")

    def is_runtime_ready(self) -> bool:
        """Return whether the required RVC runtime files are present."""
        return self.runtime_info["ready"]

    def _patch_pyav_compat(self) -> None:
        """
        Patch PyAV open-mode compatibility for the bundled RVC code.

        The official RVC copy calls `av.open(..., "rb")` / `"wb"`, but newer
        PyAV versions expect `"r"` / `"w"`.
        """
        try:
            import av
        except Exception:
            return

        if getattr(av.open, "_music_ai_patched", False):
            return

        original_open = av.open

        def patched_open(*args, **kwargs):
            if len(args) >= 2 and isinstance(args[1], str):
                mode = args[1]
                if mode == "rb":
                    args = (args[0], "r", *args[2:])
                elif mode == "wb":
                    args = (args[0], "w", *args[2:])
            elif "mode" in kwargs and isinstance(kwargs["mode"], str):
                if kwargs["mode"] == "rb":
                    kwargs["mode"] = "r"
                elif kwargs["mode"] == "wb":
                    kwargs["mode"] = "w"
            return original_open(*args, **kwargs)

        patched_open._music_ai_patched = True
        av.open = patched_open

    def _patch_rvc_audio_loader(self) -> None:
        """
        Replace the bundled fragile PyAV-based loader with a stable local one.

        The official project code breaks against newer PyAV in two places:
        - mode `"rb"` / `"wb"`
        - writing `ostream.channels`
        We bypass both by loading/resampling audio ourselves.
        """
        try:
            import librosa
            import soundfile as sf
            import infer.lib.audio as rvc_audio
            import infer.modules.vc.modules as rvc_modules
        except Exception:
            return

        if getattr(rvc_audio.load_audio, "_music_ai_patched", False):
            return

        def patched_load_audio(file, sr):
            if isinstance(file, tuple) and len(file) >= 2:
                input_sr, audio = file[0], file[1]
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim == 2:
                    audio = np.mean(audio, axis=-1)
                if np.issubdtype(audio.dtype, np.integer):
                    max_int = np.iinfo(audio.dtype).max
                    if max_int > 0:
                        audio = audio.astype(np.float32) / max_int
                if int(input_sr) != int(sr):
                    audio = librosa.resample(audio, orig_sr=int(input_sr), target_sr=int(sr))
                return np.asarray(audio, dtype=np.float32).flatten()

            path = str(file).strip().strip('"').strip()
            if not os.path.exists(path):
                raise RuntimeError(
                    "You input a wrong audio path that does not exists, please fix it!"
                )

            audio, input_sr = sf.read(path, always_2d=False)
            audio = np.asarray(audio)
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            if np.issubdtype(audio.dtype, np.integer):
                max_int = np.iinfo(audio.dtype).max
                if max_int > 0:
                    audio = audio.astype(np.float32) / max_int
            else:
                audio = audio.astype(np.float32)

            if int(input_sr) != int(sr):
                audio = librosa.resample(audio, orig_sr=int(input_sr), target_sr=int(sr))
            return np.asarray(audio, dtype=np.float32).flatten()

        patched_load_audio._music_ai_patched = True
        rvc_audio.load_audio = patched_load_audio
        rvc_modules.load_audio = patched_load_audio

    @contextmanager
    def _runtime_context(self, model_info: Optional[Dict] = None):
        """Prepare cwd, env vars, and import path for the official RVC runtime."""
        previous_cwd = Path.cwd()
        previous_weight_root = os.environ.get("weight_root")
        previous_uvr5_root = os.environ.get("weight_uvr5_root")
        previous_index_root = os.environ.get("index_root")
        previous_rmvpe_root = os.environ.get("rmvpe_root")
        previous_hubert_root = os.environ.get("hubert_root")
        sys_path_added = False

        try:
            if str(self.runtime_root) not in sys.path:
                sys.path.insert(0, str(self.runtime_root))
                sys_path_added = True

            os.chdir(self.runtime_root)

            model_parent = None
            index_parent = None
            if model_info is not None:
                model_parent = str(Path(model_info["model_path"]).parent)
                if model_info.get("index_path"):
                    index_parent = str(Path(model_info["index_path"]).parent)

            os.environ["weight_root"] = model_parent or self.runtime_info["weight_root"]
            os.environ["weight_uvr5_root"] = self.runtime_info["weight_root"].replace(
                "assets/weights", "assets/uvr5_weights"
            )
            os.environ["index_root"] = index_parent or self.runtime_info["index_root"]
            os.environ["rmvpe_root"] = self.runtime_info["rmvpe_root"]
            os.environ["hubert_root"] = self.runtime_info["hubert_root"]
            self._patch_pyav_compat()
            self._patch_rvc_audio_loader()

            yield
        finally:
            os.chdir(previous_cwd)
            if previous_weight_root is None:
                os.environ.pop("weight_root", None)
            else:
                os.environ["weight_root"] = previous_weight_root
            if previous_uvr5_root is None:
                os.environ.pop("weight_uvr5_root", None)
            else:
                os.environ["weight_uvr5_root"] = previous_uvr5_root
            if previous_index_root is None:
                os.environ.pop("index_root", None)
            else:
                os.environ["index_root"] = previous_index_root
            if previous_rmvpe_root is None:
                os.environ.pop("rmvpe_root", None)
            else:
                os.environ["rmvpe_root"] = previous_rmvpe_root
            if previous_hubert_root is None:
                os.environ.pop("hubert_root", None)
            else:
                os.environ["hubert_root"] = previous_hubert_root
            if sys_path_added:
                sys.path.remove(str(self.runtime_root))

    def _ensure_runtime_loaded(self, model_info: Dict) -> bool:
        """Load the official RVC VC object."""
        try:
            with self._runtime_context(model_info):
                from dotenv import load_dotenv
                from configs.config import Config
                from infer.modules.vc.modules import VC

                load_dotenv()
                config = Config()
                config.device = self.device
                self.config = config
                self.vc = VC(config)
                self.vc.get_vc(Path(model_info["model_path"]).name)
            return True
        except Exception as exc:
            self.last_error = f"RVC运行时加载失败: {exc}"
            print(f"  ⚠ {self.last_error}")
            self.vc = None
            self.config = None
            return False

    def load_model(self, model_path: str) -> bool:
        """Validate and load a real RVC model through the official runtime."""
        model_info = inspect_user_model(model_path)
        self.current_model_info = model_info
        self.last_error = None

        if not model_info["valid"]:
            self.last_error = model_info["reason"]
            print(f"  ⚠ 用户RVC模型不可用: {self.last_error}")
            return False

        if not self.is_runtime_ready():
            missing = ", ".join(self.runtime_info["missing_files"])
            self.last_error = f"RVC运行时未就绪: 缺少 {missing}"
            print(f"  ⚠ {self.last_error}")
            return False

        if self._ensure_runtime_loaded(model_info):
            print("  ✓ 真实RVC模型已加载")
            return True

        return False

    def convert(
        self,
        audio_path: str,
        user_model_path: str = None,
        f0_shift: int = 0,
        f0_method: str = "rmvpe",
    ):
        """
        Convert voice with RVC.

        Returns converted audio when runtime/model are ready.
        """
        if user_model_path and (
            self.current_model_info is None
            or self.current_model_info.get("model_path") != str(Path(user_model_path))
        ):
            self.load_model(user_model_path)

        if self.vc is None or self.current_model_info is None:
            return None

        try:
            with self._runtime_context(self.current_model_info):
                info, wav_opt = self.vc.vc_single(
                    0,
                    audio_path,
                    f0_shift,
                    None,
                    f0_method,
                    self.current_model_info.get("index_path") or "",
                    None,
                    0.66,
                    3,
                    0,
                    1.0,
                    0.33,
                )
        except Exception as exc:
            self.last_error = f"RVC推理失败: {exc}"
            print(f"  ⚠ {self.last_error}")
            return None

        if not wav_opt or wav_opt[0] is None or wav_opt[1] is None:
            self.last_error = f"RVC推理未返回音频: {info}"
            print(f"  ⚠ {self.last_error}")
            return None

        sample_rate, audio = wav_opt
        if sample_rate <= 0:
            self.last_error = f"RVC返回了非法采样率: {sample_rate}"
            print(f"  ⚠ {self.last_error}")
            return None

        audio = np.asarray(audio)
        if audio.ndim == 1:
            return audio.astype(np.float32), int(sample_rate)
        return audio.squeeze().astype(np.float32), int(sample_rate)


class RVCTrainer:
    """Strict RVC trainer stub."""

    def __init__(self, model_dir: str = "models/RVC1006Nvidia"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = self.project_root / model_dir
        self.runtime_info = inspect_rvc_runtime(self.model_dir)

    def is_ready(self) -> bool:
        """Return whether real training support exists."""
        return False

    def train(
        self,
        audio_path: str,
        user_id: str,
        epochs: int = 100,
        mode: str = "quick",
    ) -> str:
        """Refuse to generate fake user checkpoints."""
        raise RuntimeError(
            "当前项目未接入真实RVC训练流程，已停止保存伪造 `.pth` 用户模型。"
            " 请先准备完整 `models/RVC1006Nvidia` 运行时，并手动放入训练好的用户模型。"
        )
