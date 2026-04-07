"""
模型验证脚本
检查方案A架构所需的所有模型是否已下载并可用
"""
import os
import importlib.util
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.voice_cloning.rvc_runtime import (
    discover_runtime_models,
    discover_user_models,
    inspect_rvc_runtime,
)

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_status(message, status):
    """打印状态"""
    status_symbol = "✓" if status else "✗"
    print(f"  {status_symbol} {message}")

def check_directory(path, name):
    """检查目录是否存在"""
    if path.exists() and path.is_dir():
        files = list(path.rglob("*"))
        size = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
        print_status(f"{name}: 存在 ({len(files)} 个文件, {size:.1f} MB)", True)
        return True
    else:
        print_status(f"{name}: 不存在", False)
        return False

def check_file(path, name, min_size_mb=0):
    """检查文件是否存在"""
    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb >= min_size_mb:
            print_status(f"{name}: 存在 ({size_mb:.1f} MB)", True)
            return True
        else:
            print_status(f"{name}: 存在但太小 ({size_mb:.1f} MB, 需要至少 {min_size_mb} MB)", False)
            return False
    else:
        print_status(f"{name}: 不存在", False)
        return False

def verify_ace_step():
    """验证ACE-Step模型"""
    print_header("1. ACE-Step 模型验证")

    models_dir = project_root / "models"
    ace_step_path = models_dir / "Ace-Step1.5"

    all_ok = True

    # 检查主目录
    all_ok &= check_directory(ace_step_path, "ACE-Step主目录")

    # 检查关键文件
    if ace_step_path.exists():
        key_files = [
            (ace_step_path / "acestep-v15-turbo" / "model.safetensors", "ACE-Step主模型", 1000),  # 至少1GB
            (ace_step_path / "acestep-v15-turbo" / "config.json", "ACE-Step配置文件", 0.001),  # 配置文件很小
            (ace_step_path / "acestep-v15-turbo" / "silence_latent.pt", "ACE-Step静音潜在向量", 1),  # 至少1MB
            (ace_step_path / "acestep-5Hz-lm-1.7B" / "model.safetensors", "ACE-Step语言模型", 1000),  # 至少1GB
            (ace_step_path / "Qwen3-Embedding-0.6B" / "model.safetensors", "Qwen3嵌入模型", 500),  # 至少500MB
            (ace_step_path / "vae" / "diffusion_pytorch_model.safetensors", "VAE模型", 100),  # 至少100MB
        ]

        for file_path, description, min_size in key_files:
            all_ok &= check_file(file_path, description, min_size_mb=min_size)

    official_runtime_installed = importlib.util.find_spec("acestep") is not None
    print_status("ACE-Step官方推理包(acestep): 已安装" if official_runtime_installed else "ACE-Step官方推理包(acestep): 未安装", official_runtime_installed)
    if not official_runtime_installed:
        print("  提示: 当前项目可加载本地轻量包装器，但要更稳定地产生真实演唱，建议安装官方 ACE-Step 推理包。")

    return all_ok

def verify_demucs():
    """验证Demucs模型"""
    print_header("2. Demucs 模型验证")

    models_dir = project_root / "models"
    demucs_path = models_dir / "demucs"
    
    all_ok = False

    # 检查本地模型目录
    if demucs_path.exists():
        print_status("Demucs本地目录: 存在", True)
        
        # 检查是否有.th文件（Demucs模型文件）
        th_files = list(demucs_path.glob("*.th"))
        if len(th_files) >= 4:
            total_size = sum(f.stat().st_size for f in th_files) / (1024 * 1024)
            print_status(f"Demucs模型文件: 找到 {len(th_files)} 个文件 ({total_size:.1f} MB)", True)
            all_ok = True
        else:
            print_status(f"Demucs模型文件: 不足 ({len(th_files)} 个文件，需要至少4个)", False)
            
        # 也检查缓存目录
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_demucs_master"
        if cache_dir.exists():
            print_status("Demucs缓存目录: 存在", True)
        else:
            print_status("Demucs缓存目录: 不存在（首次运行时会自动下载）", True)
            print("  提示: Demucs会在首次使用时自动下载，无需手动操作")
    else:
        print_status("Demucs本地目录: 不存在", False)
        
        # 检查缓存目录
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_demucs_master"
        if cache_dir.exists():
            print_status("Demucs缓存目录: 存在", True)
            all_ok = check_directory(cache_dir / "htdemucs", "htdemucs模型")
        else:
            print_status("Demucs缓存目录: 不存在（首次运行时会自动下载）", False)
            print("  提示: Demucs会在首次使用时自动下载，无需手动操作")

    return all_ok

def verify_rvc():
    """验证RVC模型"""
    print_header("3. RVC 模型验证")

    models_dir = project_root / "models"
    rvc_pretrained_path = models_dir / "rvc_pretrained"
    runtime_info = inspect_rvc_runtime(models_dir / "RVC1006Nvidia")
    discovery = discover_user_models(models_dir / "user_voices")
    runtime_models = discover_runtime_models(models_dir / "RVC1006Nvidia")

    all_ok = True

    # 检查RVC预训练目录
    all_ok &= check_directory(rvc_pretrained_path, "RVC预训练目录")

    # 检查关键文件
    if rvc_pretrained_path.exists():
        key_files = [
            ("D48k.pth", "RVC判别器 (48kHz)", 50),
            ("G48k.pth", "RVC生成器 (48kHz)", 50),
            ("hubert_base.pt", "HuBERT内容编码器", 100),  # 实际大小约181MB，降低要求
        ]

        for filename, description, min_size in key_files:
            file_path = rvc_pretrained_path / filename
            all_ok &= check_file(file_path, description, min_size_mb=min_size)

    # 检查可选的RMVPE模型
    rmvpe_path = rvc_pretrained_path / "rmvpe.pt"
    if rmvpe_path.exists():
        check_file(rmvpe_path, "RMVPE音高提取器 (可选)", min_size_mb=10)
    else:
        print_status("RMVPE音高提取器: 不存在（可选）", True)

    # 检查RVC运行时代码
    if runtime_info["ready"]:
        runtime_root = Path(runtime_info["repo_path"])
        print_status(f"RVC1006Nvidia运行时代码: 可用 ({runtime_root})", True)
    else:
        missing = ", ".join(runtime_info["missing_files"])
        print_status(f"RVC1006Nvidia运行时代码: 不可用（缺少 {missing}）", False)
        if runtime_info["partial_clone"]:
            print("  提示: 当前目录像是一次未完成的 clone，只留下了 `.git`。")
        all_ok = False

    if runtime_models["valid_models"]:
        print_status(f"官方RVC示例音色: 找到 {len(runtime_models['valid_models'])} 个", True)
        default_model_path = runtime_models.get("default_model_path")
        if default_model_path:
            print(f"    默认示例音色: {Path(default_model_path).name}")
    else:
        print_status("官方RVC示例音色: 未找到", False)

    # 检查真实用户模型
    if discovery["valid_models"]:
        print_status(f"真实RVC用户模型: 找到 {len(discovery['valid_models'])} 个", True)
        for info in discovery["valid_models"][:3]:
            model_name = Path(info["model_path"]).name
            print(f"    - {model_name}")
    else:
        print_status("真实RVC用户模型: 未找到", False)
        all_ok = False

    if discovery["invalid_models"]:
        print("  当前无效的 `.pth` 文件:")
        for info in discovery["invalid_models"][:3]:
            model_name = Path(info["model_path"]).name
            print(f"    - {model_name}: {info['reason']}")

    return all_ok

def verify_models():
    """验证所有模型"""
    print("\n" + "=" * 60)
    print("  AI音乐生成项目 - 模型验证")
    print("  方案A架构: ACE-Step + SVC")
    print("=" * 60)

    results = {
        "ACE-Step": verify_ace_step(),
        "Demucs": verify_demucs(),
        "RVC": verify_rvc(),
    }

    # 打印总结
    print_header("验证总结")

    for name, result in results.items():
        status_text = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status_text}")

    # 总体状态
    all_ok = all(results.values())
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✓ 所有必需模型验证通过！")
        print("  你可以开始使用Pipeline生成歌曲了。")
    else:
        print("  ✗ 部分模型验证失败")
        print("  请根据上述提示补齐真实用户RVC模型，或先使用官方示例音色测试。")
    print("=" * 60 + "\n")

    return all_ok

def test_model_loading():
    """测试模型加载"""
    print_header("测试模型加载")

    try:
        # 测试ACE-Step
        print("\n  测试ACE-Step加载...")
        from src.music_generation.ace_step_wrapper import ACEStepWrapper
        ace = ACEStepWrapper(str(project_root / "models" / "Ace-Step1.5"))
        if ace.is_ready():
            print_status("ACE-Step: 加载成功", True)
        else:
            print_status("ACE-Step: 加载失败", False)

    except Exception as e:
        print_status(f"ACE-Step: 加载失败 - {e}", False)

    try:
        # 测试Demucs
        print("\n  测试Demucs加载...")
        from src.preprocessing.vocal_separator_demucs import VocalSeparatorDemucs
        separator = VocalSeparatorDemucs()
        if separator.is_ready():
            print_status("Demucs: 加载成功", True)
        else:
            print_status("Demucs: 加载失败", False)

    except Exception as e:
        print_status(f"Demucs: 加载失败 - {e}", False)

    try:
        # 测试RVC
        print("\n  测试RVC加载...")
        from src.voice_conversion.svc_converter import SVCConverter
        converter = SVCConverter()
        if converter.is_ready():
            print_status("RVC: 加载成功", True)
        else:
            print_status("RVC: 加载失败", False)

    except Exception as e:
        print_status(f"RVC: 加载失败 - {e}", False)

if __name__ == "__main__":
    # 验证模型
    all_ok = verify_models()
    runtime_ready = inspect_rvc_runtime(project_root / "models" / "RVC1006Nvidia")["ready"]
    runtime_models = discover_runtime_models(project_root / "models" / "RVC1006Nvidia")

    # 如果核心模型可用，继续测试加载；即使没有用户专属RVC模型，也可先测试官方示例音色。
    if all_ok or runtime_ready or runtime_models["valid_models"]:
        print("\n核心模型文件已就绪，开始测试模型加载...")
        test_model_loading()
    else:
        print("\n部分模型缺失，跳过加载测试。")
        print("请先下载缺失的模型，然后重新运行此脚本。")
