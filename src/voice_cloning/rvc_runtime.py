"""
RVC runtime and checkpoint validation helpers.

This module is intentionally strict: it does not treat placeholder `.pth` files
as real RVC voice models.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


REQUIRED_RUNTIME_FILES = (
    "configs/config.py",
    "infer/modules/vc/modules.py",
)

REAL_USER_MODEL_REQUIRED_KEYS = {"weight", "config"}
PLACEHOLDER_MODEL_KEYS = {
    "audio_sample",
    "epochs",
    "f0_stats",
    "hubert_features",
    "mode",
    "sr",
    "user_id",
}


def get_project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def resolve_rvc_runtime_root(repo_path: Optional[Path] = None) -> Path:
    """Resolve the actual RVC runtime root, including common nested layouts."""
    project_root = get_project_root()
    repo_path = Path(repo_path) if repo_path else project_root / "models" / "RVC1006Nvidia"
    if not repo_path.is_absolute():
        repo_path = project_root / repo_path

    direct_candidates = [repo_path, repo_path / "RVC1006Nvidia"]
    nested_candidates: List[Path] = []
    if repo_path.exists():
        nested_candidates = [path for path in repo_path.iterdir() if path.is_dir()]

    for candidate in direct_candidates + nested_candidates:
        if all((candidate / rel).exists() for rel in REQUIRED_RUNTIME_FILES):
            return candidate

    return repo_path


def inspect_rvc_runtime(repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """Inspect whether a local RVC runtime repository is actually usable."""
    requested_path = Path(repo_path) if repo_path else get_project_root() / "models" / "RVC1006Nvidia"
    if not requested_path.is_absolute():
        requested_path = get_project_root() / requested_path
    repo_path = resolve_rvc_runtime_root(requested_path)

    missing_files = [rel for rel in REQUIRED_RUNTIME_FILES if not (repo_path / rel).exists()]
    git_dir_exists = (repo_path / ".git").exists()
    weight_root = repo_path / "assets" / "weights"
    index_root = repo_path / "logs"
    hubert_root = repo_path / "assets" / "hubert"
    rmvpe_root = repo_path / "assets" / "rmvpe"
    config_json_path = repo_path / "configs" / "config.json"

    default_model_path = None
    default_index_path = None
    if config_json_path.exists():
        try:
            config_data = json.loads(config_json_path.read_text(encoding="utf-8"))
            raw_model_path = config_data.get("pth_path")
            raw_index_path = config_data.get("index_path")
            if raw_model_path:
                default_model_path = str((repo_path / raw_model_path).resolve())
            if raw_index_path:
                default_index_path = str((repo_path / raw_index_path).resolve())
        except Exception:
            default_model_path = None
            default_index_path = None

    return {
        "requested_path": str(requested_path),
        "repo_path": str(repo_path),
        "exists": requested_path.exists(),
        "ready": not missing_files,
        "missing_files": missing_files,
        "git_dir_exists": git_dir_exists,
        "partial_clone": bool(git_dir_exists and missing_files),
        "weight_root": str(weight_root),
        "index_root": str(index_root),
        "hubert_root": str(hubert_root),
        "rmvpe_root": str(rmvpe_root),
        "config_json_path": str(config_json_path),
        "default_model_path": default_model_path,
        "default_index_path": default_index_path,
    }


def _safe_torch_load(model_path: Path) -> Any:
    """Load a PyTorch checkpoint in a version-tolerant way."""
    try:
        return torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location="cpu")


def _find_sidecar_index(model_path: Path) -> Optional[str]:
    """Find a sidecar `.index` file next to the given model if present."""
    direct_match = model_path.with_suffix(".index")
    if direct_match.exists():
        return str(direct_match)

    stem_prefix = model_path.stem
    matches = sorted(model_path.parent.glob(f"{stem_prefix}*.index"))
    if matches:
        return str(matches[0])

    return None


def _find_runtime_index_for_model(model_path: Path, runtime_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Find a matching runtime `.index` file for an RVC model."""
    runtime_info = runtime_info or inspect_rvc_runtime()
    index_root = Path(runtime_info["index_root"])
    if not index_root.exists():
        return None

    stem_prefix = model_path.stem
    matches = sorted(index_root.rglob(f"*{stem_prefix}*.index"))
    if matches:
        return str(matches[0])
    return None


def inspect_user_model(model_path: Path | str) -> Dict[str, Any]:
    """Inspect a user voice checkpoint and classify whether it is a real RVC model."""
    model_path = Path(model_path)

    info: Dict[str, Any] = {
        "model_path": str(model_path),
        "exists": model_path.exists(),
        "valid": False,
        "model_role": "missing",
        "reason": "文件不存在",
        "checkpoint_keys": [],
        "index_path": None,
    }

    if not model_path.exists() or not model_path.is_file():
        return info

    runtime_info = inspect_rvc_runtime()
    info["index_path"] = _find_sidecar_index(model_path) or _find_runtime_index_for_model(
        model_path, runtime_info
    )

    try:
        checkpoint = _safe_torch_load(model_path)
    except Exception as exc:
        info["model_role"] = "unreadable"
        info["reason"] = f"无法读取检查点: {exc}"
        return info

    if isinstance(checkpoint, dict):
        keys = sorted(str(key) for key in checkpoint.keys())
        info["checkpoint_keys"] = keys
        key_set = set(keys)

        if REAL_USER_MODEL_REQUIRED_KEYS.issubset(key_set):
            info["valid"] = True
            info["model_role"] = "rvc_user_model"
            info["reason"] = "检测到真实RVC用户模型权重"
            return info

        if PLACEHOLDER_MODEL_KEYS & key_set:
            info["model_role"] = "placeholder"
            info["reason"] = "这是项目之前伪造的占位 `.pth`，不是真实RVC用户模型"
            return info

        if "model" in key_set and "weight" not in key_set:
            info["model_role"] = "pretrained_backbone"
            info["reason"] = "这是RVC预训练骨架权重，不是用户音色模型"
            return info

        info["model_role"] = "unknown_dict"
        info["reason"] = "检查点缺少真实RVC用户模型常见键 `weight` 和 `config`"
        return info

    if isinstance(checkpoint, torch.nn.Module):
        info["model_role"] = "torch_module"
        info["reason"] = "检测到PyTorch模块对象，但当前项目未接入其真实推理入口"
        return info

    info["model_role"] = type(checkpoint).__name__
    info["reason"] = "检查点类型不是项目可识别的RVC用户模型"
    return info


def discover_user_models(user_voice_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Scan `models/user_voices` and separate valid and invalid model files."""
    project_root = get_project_root()
    user_voice_dir = Path(user_voice_dir) if user_voice_dir else project_root / "models" / "user_voices"
    if not user_voice_dir.is_absolute():
        user_voice_dir = project_root / user_voice_dir

    valid_models: List[Dict[str, Any]] = []
    invalid_models: List[Dict[str, Any]] = []

    if user_voice_dir.exists():
        for model_path in sorted(user_voice_dir.glob("*.pth")):
            info = inspect_user_model(model_path)
            if info["valid"]:
                valid_models.append(info)
            else:
                invalid_models.append(info)

    return {
        "directory": str(user_voice_dir),
        "exists": user_voice_dir.exists(),
        "valid_models": valid_models,
        "invalid_models": invalid_models,
    }


def discover_runtime_models(repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """Scan the official RVC runtime `assets/weights` directory for usable models."""
    runtime_info = inspect_rvc_runtime(repo_path)
    weight_root = Path(runtime_info["weight_root"])
    valid_models: List[Dict[str, Any]] = []
    invalid_models: List[Dict[str, Any]] = []

    if weight_root.exists():
        default_model_path = runtime_info.get("default_model_path")
        for model_path in sorted(weight_root.glob("*.pth")):
            info = inspect_user_model(model_path)
            info["source"] = "runtime_weights"
            info["is_default"] = str(model_path.resolve()) == default_model_path
            if info["valid"]:
                valid_models.append(info)
            else:
                invalid_models.append(info)

    return {
        "directory": str(weight_root),
        "exists": weight_root.exists(),
        "valid_models": valid_models,
        "invalid_models": invalid_models,
        "default_model_path": runtime_info.get("default_model_path"),
        "default_index_path": runtime_info.get("default_index_path"),
    }


def find_user_model_for_user(user_id: str, user_voice_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Find a valid user model matching a user id."""
    user_id = str(user_id).lower()
    discovery = discover_user_models(user_voice_dir)

    exact_matches: List[Dict[str, Any]] = []
    partial_matches: List[Dict[str, Any]] = []
    for info in discovery["valid_models"]:
        model_name = Path(info["model_path"]).stem.lower()
        if model_name in {
            user_id,
            f"user_{user_id}",
            f"user_{user_id}_voice",
            f"user_{user_id}_rvc",
        }:
            exact_matches.append(info)
        elif user_id in model_name:
            partial_matches.append(info)

    if exact_matches:
        return exact_matches[0]
    if partial_matches:
        return partial_matches[0]
    if len(discovery["valid_models"]) == 1:
        return discovery["valid_models"][0]
    return None


def find_runtime_model_by_name(model_name: str, repo_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Find a runtime bundled voice model by filename stem or exact name."""
    discovery = discover_runtime_models(repo_path)
    model_name = str(model_name).lower()
    for info in discovery["valid_models"]:
        path = Path(info["model_path"])
        if path.name.lower() == model_name or path.stem.lower() == model_name:
            return info
    return None


def get_default_runtime_model(repo_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the default runtime model configured by the official RVC project."""
    discovery = discover_runtime_models(repo_path)
    for info in discovery["valid_models"]:
        if info.get("is_default"):
            return info
    return None


def select_rvc_model(
    user_id: Optional[str] = None,
    user_voice_dir: Optional[Path] = None,
    preferred_model_name: Optional[str] = None,
    preferred_model_path: Optional[str] = None,
    allow_runtime_default: bool = False,
) -> Optional[Dict[str, Any]]:
    """Select a model from explicit env vars, user voices, or optional runtime default."""
    if preferred_model_path:
        preferred_info = inspect_user_model(preferred_model_path)
        if preferred_info["valid"]:
            preferred_info["source"] = "preferred_path"
            return preferred_info

    if preferred_model_name:
        preferred_runtime_info = find_runtime_model_by_name(preferred_model_name)
        if preferred_runtime_info is not None:
            preferred_runtime_info["requested_name"] = preferred_model_name
            return preferred_runtime_info

    explicit_model_path = os.getenv("MUSIC_AI_RVC_MODEL_PATH")
    if explicit_model_path:
        explicit_info = inspect_user_model(explicit_model_path)
        if explicit_info["valid"]:
            explicit_info["source"] = "explicit_path"
            return explicit_info

    explicit_model_name = os.getenv("MUSIC_AI_RVC_MODEL_NAME")
    if explicit_model_name:
        explicit_runtime_info = find_runtime_model_by_name(explicit_model_name)
        if explicit_runtime_info is not None:
            return explicit_runtime_info

    if user_id:
        user_model = find_user_model_for_user(user_id, user_voice_dir=user_voice_dir)
        if user_model is not None:
            user_model["source"] = "user_voices"
            return user_model

    if allow_runtime_default:
        runtime_default = get_default_runtime_model()
        if runtime_default is not None:
            return runtime_default

    return None
