"""检查Python环境和依赖"""
import sys

def check_python_version():
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ 错误: 需要Python 3.x")
        return False
    
    if version.minor < 9:
        print("❌ 错误: Python版本过低，需要 >= 3.9")
        return False
    
    if version.minor > 11:
        print("⚠️  警告: Python 3.12+ 可能存在兼容性问题")
        print("   推荐使用 Python 3.10")
    
    if version.minor == 10:
        print("✅ 完美! Python 3.10 是推荐版本")
        return True
    
    if version.minor in [9, 11]:
        print("✅ 可用版本")
        return True
    
    return True

def check_dependencies():
    """检查关键依赖"""
    deps = {
        "torch": "2.1.0",
        "transformers": "4.36.0",
        "librosa": "0.10.1",
        "fastapi": "0.109.0"
    }
    
    print("\n检查依赖库:")
    all_ok = True
    
    for package, expected in deps.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            status = "✅" if version.startswith(expected.split(".")[0]) else "⚠️"
            print(f"{status} {package}: {version} (期望: {expected})")
        except ImportError:
            print(f"❌ {package}: 未安装")
            all_ok = False
    
    return all_ok

def check_cuda():
    """检查CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ CUDA可用: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
        else:
            print("\n⚠️  CUDA不可用，将使用CPU (速度较慢)")
    except ImportError:
        print("\n❌ PyTorch未安装")

if __name__ == "__main__":
    print("=" * 60)
    print("AI音乐生成系统 - 环境检查")
    print("=" * 60)
    
    if not check_python_version():
        sys.exit(1)
    
    check_dependencies()
    check_cuda()
    
    print("\n" + "=" * 60)
    print("环境检查完成")
    print("=" * 60)
