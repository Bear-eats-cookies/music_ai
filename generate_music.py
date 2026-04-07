#!/usr/bin/env python3
"""
完整音乐生成脚本 - 使用ACE-Step生成真实音乐
"""
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import soundfile as sf
import torch

from src.music_generation.ace_step_wrapper import ACEStepWrapper


def generate_song(
    prompt: str,
    lyrics: str,
    duration: int = 90,
    output_path: str = "generated_song.wav",
    seed: int = 42,
):
    """生成完整歌曲。"""
    print("\n" + "=" * 60)
    print("AI音乐生成系统 - ACE-Step v1.5")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("\n⚠ 使用CPU运行（速度较慢）")

    print("\n加载ACE-Step模型...")
    ace = ACEStepWrapper("models/Ace-Step1.5")

    if not ace.is_ready():
        print("✗ ACE-Step模型未就绪")
        return None

    print("✓ ACE-Step模型已加载")
    print("\n开始生成音乐...")
    print(f"  风格: {prompt}")
    print(f"  时长: {duration}秒")
    print(f"  歌词: {lyrics[:50]}...")

    audio = ace.generate_music(
        prompt=prompt,
        duration=duration,
        lyrics=lyrics,
        seed=seed,
    )

    if audio is None:
        print("✗ 音乐生成失败")
        return None

    sf.write(output_path, audio, 48000)

    print("\n✓ 音乐生成成功!")
    print(f"  输出文件: {output_path}")
    print(f"  文件大小: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  时长: {len(audio) / 48000:.2f} 秒")

    return output_path


if __name__ == "__main__":
    prompt = "emotional Chinese pop ballad with vocals, 80 BPM, piano and strings, heartfelt singing"
    lyrics = """夜空中最亮的星
能否听清
那仰望的人
心底的孤独和叹息

夜空中最亮的星
能否记起
曾与我同行
消失在风里的身影"""

    generate_song(
        prompt=prompt,
        lyrics=lyrics,
        duration=90,
        output_path="my_generated_song.wav",
        seed=42,
    )
