"""
PokieTicker 一键启动脚本
同时启动后端 FastAPI 服务和前端开发服务器
"""
import subprocess
import sys
import os
import threading
import time


def run_backend():
    """启动后端服务"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "backend.api.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
        check=True
    )


def run_frontend():
    """启动前端服务"""
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    subprocess.Popen(
        ["npm.cmd", "run", "dev"],
        cwd=frontend_dir
    )


if __name__ == "__main__":
    print("=" * 50)
    print("启动 PokieTicker 项目...")
    print("=" * 50)

    # 先检查依赖
    print("\n[1/2] 检查后端依赖...")
    try:
        import fastapi, uvicorn
        print("  后端依赖 OK")
    except ImportError as e:
        print(f"  错误: {e}")
        print("  请运行: pip install -r requirements.txt")
        sys.exit(1)

    print("\n[2/2] 检查前端依赖...")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("  正在安装前端依赖 (首次运行较慢)...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    print("  前端依赖 OK")

    print("\n启动服务:")
    print("  - 后端: http://127.0.0.1:8000")
    print("  - 前端: http://127.0.0.1:5173")
    print("\n按 Ctrl+C 停止所有服务\n")

    # 并行启动
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)

    backend_thread.start()
    time.sleep(1)  # 后端先启动
    frontend_thread.start()

    try:
        frontend_thread.join()
    except KeyboardInterrupt:
        print("\n已停止所有服务")
        sys.exit(0)
