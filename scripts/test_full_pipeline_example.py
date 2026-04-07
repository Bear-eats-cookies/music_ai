#!/usr/bin/env python3
"""
Run the full ACE-Step -> Demucs -> RVC pipeline with an official example voice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MusicAIPipeline
from src.voice_cloning.rvc_runtime import (
    discover_runtime_models,
    find_runtime_model_by_name,
    get_default_runtime_model,
)


DEFAULT_LYRICS = """夜空中最亮的星
能否听清
那仰望的人
心底的孤独和叹息"""


def default_input_path() -> Path:
    """Pick a practical local input sample for smoke testing."""
    candidates = (
        PROJECT_ROOT / "data" / "outputs" / "song_65348_separated" / "song_65348_original_vocals.wav",
        PROJECT_ROOT / "data" / "outputs" / "rvc_example_test.wav",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full music pipeline with an official RVC example voice."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=str(default_input_path()),
        help="Input user/sample audio path.",
    )
    parser.add_argument(
        "--user-id",
        default="pipeline_example",
        help="Logical user id used for output folders.",
    )
    parser.add_argument(
        "--style",
        default="pop_ballad",
        help="Target music style.",
    )
    parser.add_argument(
        "--lyrics",
        default=DEFAULT_LYRICS,
        help="Lyrics used for generation.",
    )
    parser.add_argument(
        "--rvc-model-name",
        default=None,
        help="Official RVC example voice name, such as kikiV1 or keruanV1.",
    )
    parser.add_argument(
        "--list-rvc-models",
        action="store_true",
        help="List bundled official RVC example voices and exit.",
    )
    return parser.parse_args()


def print_runtime_models() -> int:
    """Print available bundled official RVC example voices."""
    discovery = discover_runtime_models(PROJECT_ROOT / "models" / "RVC1006Nvidia")
    if not discovery["valid_models"]:
        print("没有找到可用的官方 RVC 示例音色。")
        return 1

    print("可用的官方 RVC 示例音色:")
    for info in discovery["valid_models"]:
        suffix = " (default)" if info.get("is_default") else ""
        print(f"  - {Path(info['model_path']).stem}{suffix}")
    return 0


def resolve_model_name(requested_model_name: str | None) -> str | None:
    """Resolve the requested official example voice name."""
    if requested_model_name:
        model_info = find_runtime_model_by_name(requested_model_name, PROJECT_ROOT / "models" / "RVC1006Nvidia")
        if model_info is None:
            raise SystemExit(f"未找到官方 RVC 示例音色: {requested_model_name}")
        return Path(model_info["model_path"]).stem

    default_model = get_default_runtime_model(PROJECT_ROOT / "models" / "RVC1006Nvidia")
    if default_model is None:
        raise SystemExit("没有找到默认官方 RVC 示例音色，请先检查 `models/RVC1006Nvidia`。")
    return Path(default_model["model_path"]).stem


def main() -> int:
    args = parse_args()

    if args.list_rvc_models:
        return print_runtime_models()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"输入音频不存在: {input_path}")
        print("请提供 `--input`，或先准备一个可用的人声样本。")
        return 1

    model_name = resolve_model_name(args.rvc_model_name)

    print("=" * 60)
    print("完整链路测试")
    print("=" * 60)
    print(f"输入音频: {input_path}")
    print(f"用户ID: {args.user_id}")
    print(f"风格: {args.style}")
    print(f"RVC示例音色: {model_name}")
    print("=" * 60)

    pipeline = MusicAIPipeline()
    result = pipeline.run(
        audio_path=str(input_path),
        user_id=args.user_id,
        lyrics=args.lyrics,
        style=args.style,
        voice_model_name=model_name,
    )

    voice_profile = result["metadata"]["voice_profile"]
    print("\n生成完成:")
    print(f"  最终歌曲: {result['final_song_path']}")
    print(f"  原始歌曲: {result['original_song_path']}")
    print(f"  转换人声: {result['vocal_path']}")
    print(f"  伴奏: {result['instrumental_path']}")
    print(f"  实际使用RVC模型: {voice_profile.get('voice_model_path')}")
    print(f"  RVC来源: {voice_profile.get('source')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
