"""
RVC runtime wrapper.

This file no longer fabricates user models or fake conversion output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from .rvc_runtime import inspect_rvc_runtime, inspect_user_model


class RVCWrapper:
    """Thin wrapper around an external RVC runtime checkout."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.project_root = Path(__file__).resolve().parents[2]
        self.rvc_path = self.project_root / "models" / "RVC1006Nvidia"
        self.runtime_info = inspect_rvc_runtime(self.rvc_path)
        self.vc: Any = None

        print(f"  RVC路径: {self.rvc_path}")
        if self.runtime_info["ready"]:
            if str(self.rvc_path) not in sys.path:
                sys.path.insert(0, str(self.rvc_path))
            os.environ["weight_root"] = str(self.rvc_path / "assets" / "weights")
            os.environ["weight_uvr5_root"] = str(self.rvc_path / "assets" / "uvr5_weights")
            os.environ["index_root"] = str(self.rvc_path / "logs")
        else:
            missing = ", ".join(self.runtime_info["missing_files"])
            print(f"  ⚠ RVC运行时未就绪: 缺少 {missing}")

    def is_runtime_ready(self) -> bool:
        """Return whether a usable RVC runtime checkout is present."""
        return self.runtime_info["ready"]

    def load_vc(self):
        """Load the VC entrypoint from a real RVC runtime checkout."""
        if self.vc is not None:
            return self.vc

        if not self.is_runtime_ready():
            return None

        try:
            from configs.config import Config
            from infer.modules.vc.modules import VC

            config = Config()
            self.vc = VC(config)
            print("  ✓ RVC VC模块加载成功")
            return self.vc
        except Exception as exc:
            print(f"  ⚠ RVC运行时加载失败: {exc}")
            return None

    def train_voice(self, audio_path: str, user_id: str, mode: str = "quick") -> str:
        """
        Train a user voice model.

        Real training is not wired in this project. The old placeholder-save
        behavior is intentionally removed because it generated fake `.pth`
        checkpoints.
        """
        raise RuntimeError(
            "当前项目未接入真实RVC训练流程，已停止生成伪造 `.pth` 用户模型。"
            " 请先准备完整 `models/RVC1006Nvidia` 运行时，并手动放入训练好的用户模型。"
        )

    def convert_voice(
        self,
        input_audio: str,
        user_model_path: str,
        output_path: str,
        f0_shift: int = 0,
    ) -> str:
        """
        Convert singing voice with a real RVC runtime.

        This project does not ship a verified inference bridge yet, so this
        method refuses to fake the output.
        """
        if not self.is_runtime_ready():
            missing = ", ".join(self.runtime_info["missing_files"])
            raise RuntimeError(f"RVC运行时未就绪，缺少 {missing}")

        model_info = inspect_user_model(user_model_path)
        if not model_info["valid"]:
            raise RuntimeError(f"用户模型不可用: {model_info['reason']}")

        raise RuntimeError(
            "检测到真实RVC运行时和用户模型，但当前项目尚未接入可验证的RVC推理调用，"
            "因此不会伪造转换结果。"
        )
