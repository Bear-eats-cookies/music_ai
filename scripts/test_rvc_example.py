#!/usr/bin/env python3
"""
Smoke-test real RVC conversion with an official bundled example voice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.voice_cloning.rvc_runtime import select_rvc_model
from src.voice_conversion.svc_converter import SVCConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RVC conversion with a bundled example voice.")
    parser.add_argument(
        "--input",
        dest="input_path",
        default=str(
            PROJECT_ROOT
            / "data"
            / "outputs"
            / "song_65348_separated"
            / "song_65348_original_vocals.wav"
        ),
        help="Input vocal wav path.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(PROJECT_ROOT / "data" / "outputs" / "rvc_example_test.wav"),
        help="Converted output wav path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        print("请先提供一个分离出来的人声 wav，或修改 --input 参数。")
        return 1

    model_info = select_rvc_model(allow_runtime_default=True)
    if model_info is None:
        print("没有找到可用的 RVC 模型。")
        return 1

    print(f"使用RVC模型: {model_info['model_path']}")
    print(f"使用index: {model_info.get('index_path') or '无'}")
    print(f"输入音频: {input_path}")
    print(f"输出音频: {output_path}")

    converter = SVCConverter(model_path=model_info["model_path"])
    if not converter.is_ready():
        print(f"RVC转换器未就绪: {converter.last_error}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = converter.convert(
        str(input_path),
        {
            "user_id": "example_rvc",
            "voice_model_path": model_info["model_path"],
            "index_path": model_info.get("index_path"),
            "mode": "rvc_runtime_default",
            "source": model_info.get("source", "unknown"),
        },
        output_path=str(output_path),
        f0_shift=0,
    )
    print(f"转换完成: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
