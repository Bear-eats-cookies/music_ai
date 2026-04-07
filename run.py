"""启动脚本 - 内存优化版"""
import os
import sys
import gc
from pathlib import Path

# 添加src到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# GPU加速配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一块GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 优化显存分配

# 内存优化
os.environ['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数
os.environ['MKL_NUM_THREADS'] = '4'  # 限制MKL线程数

import torch
import uvicorn

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI音乐生成系统 - 启动中 (内存优化版)...")
    print("="*60)
    
    # 检测GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ GPU加速已启用: {gpu_name} ({gpu_memory:.1f}GB)")
        # 设置GPU内存增长策略
        torch.cuda.empty_cache()
    else:
        print("\n⚠ 未检测到GPU，将使用CPU运行（速度较慢）")
    
    # 检查系统内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"系统内存: {memory.total/1024**3:.1f}GB (可用: {memory.available/1024**3:.1f}GB)")
    except:
        pass
    
    print("\n访问地址:")
    print("  - 主页: http://localhost:8300")
    print("  - API文档: http://localhost:8300/docs")
    print("\n按 Ctrl+C 停止服务\n")
    
    # 强制垃圾回收
    gc.collect()
    
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8300,
        reload=True,
        reload_dirs=[str(project_root / "src")]
    )
