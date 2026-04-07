"""
使用 Git LFS 下载 SVS 模型
这是最可靠的下载方法
"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

def check_git_lfs():
    """检查 Git LFS 是否安装"""
    try:
        result = subprocess.run(['git', 'lfs', 'version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git LFS 已安装")
            print(f"   版本: {result.stdout.strip()}")
            return True
        else:
            print("❌ Git LFS 未安装")
            return False
    except FileNotFoundError:
        print("❌ Git 未安装")
        return False

def install_git_lfs_guide():
    """显示 Git LFS 安装指南"""
    print("\n" + "=" * 60)
    print("Git LFS 安装指南")
    print("=" * 60)
    print("\n方法 1: 使用安装包（推荐）")
    print("  1. 访问: https://git-lfs.github.com/")
    print("  2. 下载 Windows 安装包")
    print("  3. 运行安装程序")
    print("  4. 重新运行此脚本")
    
    print("\n方法 2: 使用 Chocolatey")
    print("  choco install git-lfs")
    
    print("\n方法 3: 使用 Scoop")
    print("  scoop install git-lfs")
    
    print("\n安装后运行:")
    print("  git lfs install")
    print()

def download_diffsinger():
    """下载 DiffSinger"""
    print("\n" + "=" * 60)
    print("下载 DiffSinger")
    print("=" * 60)
    
    model_path = MODELS_DIR / "diffsinger"
    
    # 如果目录已存在且有内容，询问是否覆盖
    if model_path.exists() and any(model_path.iterdir()):
        files = list(model_path.glob("*.pth")) + list(model_path.glob("*.pt"))
        if files:
            print(f"⚠️  目录已存在且包含 {len(files)} 个模型文件")
            choice = input("是否重新下载？(y/N): ").strip().lower()
            if choice != 'y':
                print("跳过下载")
                return True
    
    print("\n尝试下载 DiffSinger...")
    print("仓库: https://huggingface.co/openvpi/DiffSinger")
    print("这可能需要 10-30 分钟，取决于网络速度\n")
    
    try:
        # 切换到 models 目录
        subprocess.run(['git', 'lfs', 'install'], check=True)
        
        # 克隆仓库
        cmd = [
            'git', 'clone',
            'https://huggingface.co/openvpi/DiffSinger',
            str(model_path)
        ]
        
        print(f"执行命令: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=str(MODELS_DIR))
        
        if result.returncode == 0:
            print("\n✅ DiffSinger 下载完成！")
            return True
        else:
            print("\n❌ 下载失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 下载出错: {e}")
        return False

def download_vits_svs():
    """下载 VITS-SVS"""
    print("\n" + "=" * 60)
    print("下载 VITS-SVS")
    print("=" * 60)
    
    model_path = MODELS_DIR / "vits-svs"
    
    # 如果目录已存在且有内容，询问是否覆盖
    if model_path.exists() and any(model_path.iterdir()):
        files = list(model_path.glob("*.pth")) + list(model_path.glob("*.pt"))
        if files:
            print(f"⚠️  目录已存在且包含 {len(files)} 个模型文件")
            choice = input("是否重新下载？(y/N): ").strip().lower()
            if choice != 'y':
                print("跳过下载")
                return True
    
    print("\n尝试下载 VITS-SVS...")
    print("仓库: https://huggingface.co/PlayVoice/vits_chinese")
    print("这可能需要 10-20 分钟，取决于网络速度\n")
    
    try:
        # 克隆仓库
        cmd = [
            'git', 'clone',
            'https://huggingface.co/PlayVoice/vits_chinese',
            str(model_path)
        ]
        
        print(f"执行命令: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=str(MODELS_DIR))
        
        if result.returncode == 0:
            print("\n✅ VITS-SVS 下载完成！")
            return True
        else:
            print("\n❌ 下载失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 下载出错: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("SVS 模型 Git LFS 下载工具")
    print("=" * 60)
    
    # 检查 Git LFS
    if not check_git_lfs():
        install_git_lfs_guide()
        input("\n安装完成后按 Enter 继续...")
        if not check_git_lfs():
            print("\n❌ 仍未检测到 Git LFS，请先安装")
            return
    
    print("\n请选择操作：")
    print("1. 下载 DiffSinger (~2GB)")
    print("2. 下载 VITS-SVS (~1.5GB)")
    print("3. 下载全部")
    print("0. 退出")
    
    choice = input("\n请输入选项 (0-3): ").strip()
    
    if choice == "1":
        download_diffsinger()
    elif choice == "2":
        download_vits_svs()
    elif choice == "3":
        download_diffsinger()
        download_vits_svs()
    elif choice == "0":
        print("退出")
        return
    else:
        print("❌ 无效选项")
        return
    
    print("\n" + "=" * 60)
    print("操作完成！")
    print("=" * 60)
    
    # 验证下载
    print("\n验证模型文件：")
    
    diffsinger_path = MODELS_DIR / "diffsinger"
    if diffsinger_path.exists():
        files = list(diffsinger_path.glob("*.pth")) + list(diffsinger_path.glob("*.pt"))
        if files:
            print(f"✅ DiffSinger: 找到 {len(files)} 个模型文件")
        else:
            print("⚠️  DiffSinger: 目录存在但未找到模型文件")
    else:
        print("❌ DiffSinger: 目录不存在")
    
    vits_path = MODELS_DIR / "vits-svs"
    if vits_path.exists():
        files = list(vits_path.glob("*.pth")) + list(vits_path.glob("*.pt"))
        if files:
            print(f"✅ VITS-SVS: 找到 {len(files)} 个模型文件")
        else:
            print("⚠️  VITS-SVS: 目录存在但未找到模型文件")
    else:
        print("❌ VITS-SVS: 目录不存在")

if __name__ == "__main__":
    main()
