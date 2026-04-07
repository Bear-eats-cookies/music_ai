#!/usr/bin/env python3
"""
快速音乐生成脚本 - 一键生成AI歌曲
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.music_generation.ace_step_wrapper import ACEStepWrapper
import soundfile as sf
import torch

def quick_generate():
    """快速生成一首歌曲"""
    print("\n" + "="*60)
    print("🎵 AI音乐生成 - 快速模式")
    print("="*60)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU加速: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠ 使用CPU（速度较慢）")
    
    # 加载模型
    print("\n[1/3] 加载ACE-Step模型...")
    ace = ACEStepWrapper('models/Ace-Step1.5')
    
    if not ace.is_ready():
        print("✗ 模型加载失败")
        return
    
    print("✓ 模型加载成功")
    
    # 生成音乐
    print("\n[2/3] 生成音乐...")
    prompt = "emotional Chinese pop ballad with vocals, 80 BPM, piano and strings"
    lyrics = """夜空中最亮的星
能否听清
那仰望的人
心底的孤独和叹息"""
    
    audio = ace.generate_music(
        prompt=prompt,
        duration=60,  # 1分钟
        lyrics=lyrics,
        seed=42
    )
    
    if audio is None:
        print("✗ 生成失败")
        return
    
    # 保存
    print("\n[3/3] 保存音频...")
    output_file = "quick_song.wav"
    sf.write(output_file, audio, 48000)
    
    print(f"\n✅ 完成！")
    print(f"  输出文件: {output_file}")
    print(f"  时长: {len(audio)/48000:.1f} 秒")
    print(f"  文件大小: {Path(output_file).stat().st_size/1024/1024:.2f} MB")
    print("\n🎵 您可以播放这个文件欣赏AI生成的音乐！")
    print("="*60)

if __name__ == "__main__":
    try:
        quick_generate()
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
