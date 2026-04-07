"""启动API服务"""
import uvicorn

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI音乐生成系统 - 启动中...")
    print("="*60)
    print("\n访问地址:")
    print("  - 主页: http://localhost:8080")
    print("  - API文档: http://localhost:8080/docs")
    print("\n按 Ctrl+C 停止服务\n")
    
    # 注意：uvicorn 开启 reload 时需要传入 import string，否则会出现警告且reload不生效
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
