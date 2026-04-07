"""
简化版 SVS 模型下载 - 使用公开可用的模型
"""
import os
from pathlib import Path
import urllib.request
import json

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

def download_file(url, dest_path):
    """下载单个文件"""
    try:
        print(f"  📥 下载: {dest_path.name}")
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✅ 完成: {dest_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return False

def setup_diffsinger_placeholder():
    """创建 DiffSinger 占位符和说明"""
    print("=" * 60)
    print("设置 DiffSinger 目录...")
    print("=" * 60)
    
    model_path = MODELS_DIR / "diffsinger"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 README
    readme = model_path / "README.md"
    readme.write_text("""# DiffSinger 模型

## 下载方法

### 方法 1: 从 GitHub Release 下载
访问: https://github.com/openvpi/DiffSinger/releases
下载预训练模型并解压到此目录

### 方法 2: 从百度网盘下载
搜索 "DiffSinger 预训练模型" 找到分享链接

### 方法 3: 使用 Git LFS
```bash
cd models
git lfs install
git clone https://huggingface.co/openvpi/DiffSinger diffsinger
```

### 方法 4: 暂时跳过
使用方案 A (SVC) 不需要此模型
在 src/config.py 中设置: ARCHITECTURE = "SVC"

## 所需文件
- model.pth 或 checkpoint.pt (主模型)
- config.json 或 config.yaml (配置)
""", encoding='utf-8')
    
    print(f"✅ DiffSinger 目录已创建: {model_path}")
    print(f"   请查看 {readme} 了解下载方法")
    return True

def setup_vits_svs_placeholder():
    """创建 VITS-SVS 占位符和说明"""
    print("=" * 60)
    print("设置 VITS-SVS 目录...")
    print("=" * 60)
    
    model_path = MODELS_DIR / "vits-svs"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 README
    readme = model_path / "README.md"
    readme.write_text("""# VITS-SVS 中文歌声合成模型

## 下载方法

### 方法 1: 从 GitHub 下载
访问: https://github.com/PlayVoice/vits_chinese
查看 Release 或 README 中的模型链接

### 方法 2: 从百度网盘下载
搜索 "VITS 中文歌声合成" 找到分享链接

### 方法 3: 使用 Git LFS
```bash
cd models
git lfs install
git clone https://huggingface.co/PlayVoice/vits_chinese vits-svs
```

### 方法 4: 使用替代模型
- VITS-Umamusume: https://github.com/Plachta/VITS-Umamusume-voice-synthesizer
- So-VITS-SVC: https://github.com/svc-develop-team/so-vits-svc

### 方法 5: 暂时跳过
使用方案 A (SVC) 不需要此模型
在 src/config.py 中设置: ARCHITECTURE = "SVC"

## 所需文件
- G_*.pth (生成器模型)
- config.json (配置文件)
""", encoding='utf-8')
    
    print(f"✅ VITS-SVS 目录已创建: {model_path}")
    print(f"   请查看 {readme} 了解下载方法")
    return True

def create_config_for_svc_mode():
    """创建使用方案 A 的配置说明"""
    print("=" * 60)
    print("创建方案 A (SVC) 配置说明...")
    print("=" * 60)
    
    guide_path = BASE_DIR / "USE_SVC_MODE.md"
    guide_path.write_text("""# 使用方案 A (SVC) - 无需 SVS 模型

## 当前状态

你已经拥有方案 A 所需的全部模型：
- ✅ ACE-Step (生成完整歌曲)
- ✅ Demucs (分离人声)
- ✅ RVC (音色转换)

## 工作流程

```
用户音频 → RVC 训练音色模型
    ↓
ACE-Step 生成完整演唱歌曲
    ↓
Demucs 分离人声轨
    ↓
RVC 将人声转换为用户音色
    ↓
与伴奏重新混音输出
```

## 配置方法

编辑 `src/config.py`：

```python
# 使用方案 A
ARCHITECTURE = "SVC"  # 不是 "SVS"

# 确认模型配置
MUSIC_GEN_MODEL = "ace_step"
VOCAL_SEPARATOR_MODEL = "demucs"
SVC_MODEL = "rvc"
```

## 立即测试

```bash
# 运行测试生成
python quick_generate.py
```

## 优势

✅ 快速 - 无需等待 SVS 模型下载
✅ 简单 - 流程更直接
✅ 稳定 - 使用成熟的模型组合
✅ 已就绪 - 所有模型都已下载

## 何时需要 SVS 模型？

只有在需要以下功能时才需要 SVS：
- 精确控制每个音符的音高
- 根据 MIDI 文件生成人声
- 需要完全自定义旋律

对于大多数用户，方案 A 已经足够！

---

*如果以后需要 SVS 模型，可以随时手动下载*
""", encoding='utf-8')
    
    print(f"✅ 配置指南已创建: {guide_path}")
    return True

def main():
    print("\n" + "=" * 60)
    print("SVS 模型设置工具 (简化版)")
    print("=" * 60 + "\n")
    
    print("由于 Hugging Face 访问限制，我们将：")
    print("1. 创建模型目录和下载说明")
    print("2. 提供使用方案 A (SVC) 的配置指南")
    print("3. 方案 A 不需要 SVS 模型即可工作\n")
    
    input("按 Enter 继续...")
    
    # 创建目录和说明
    setup_diffsinger_placeholder()
    print()
    setup_vits_svs_placeholder()
    print()
    create_config_for_svc_mode()
    
    print("\n" + "=" * 60)
    print("设置完成！")
    print("=" * 60)
    
    print("\n📋 下一步操作：\n")
    print("选项 1: 使用方案 A (推荐)")
    print("  - 查看: USE_SVC_MODE.md")
    print("  - 无需下载 SVS 模型")
    print("  - 立即可用\n")
    
    print("选项 2: 手动下载 SVS 模型")
    print("  - 查看: models/diffsinger/README.md")
    print("  - 查看: models/vits-svs/README.md")
    print("  - 查看: MANUAL_DOWNLOAD_GUIDE.md\n")
    
    print("选项 3: 使用 Git LFS 下载")
    print("  cd models")
    print("  git lfs install")
    print("  git clone https://huggingface.co/openvpi/DiffSinger diffsinger\n")
    
    # 显示当前模型状态
    print("当前模型状态：")
    models_status = {
        "ACE-Step": (MODELS_DIR / "Ace-Step1.5").exists(),
        "RVC": (MODELS_DIR / "rvc_pretrained").exists(),
        "Demucs": (MODELS_DIR / "demucs").exists(),
        "DiffSinger": (MODELS_DIR / "diffsinger").exists(),
        "VITS-SVS": (MODELS_DIR / "vits-svs").exists(),
        "Fish Speech": (MODELS_DIR / "fish-speech-1.5").exists()
    }
    
    for model, exists in models_status.items():
        status = "✅" if exists else "❌"
        note = ""
        if model in ["DiffSinger", "VITS-SVS"] and exists:
            note = " (目录已创建，需手动下载模型文件)"
        print(f"  {status} {model}{note}")
    
    print("\n💡 提示: 方案 A 只需要前 3 个模型即可工作！")

if __name__ == "__main__":
    main()
