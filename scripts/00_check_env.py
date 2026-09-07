"""环境自检脚本：确认 PyTorch CUDA、依赖版本是否就绪。

用法: python scripts/00_check_env.py
"""
import sys


def main():
    print(f"Python: {sys.version}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # 50 系显卡（sm_120）必须 CUDA 12.8+ 版 PyTorch
            cap = torch.cuda.get_device_capability(0)
            print(f"计算能力: sm_{cap[0]}{cap[1]}")
    except ImportError:
        print("[缺失] torch 未安装")
        sys.exit(1)

    for mod in ("cv2", "rasterio", "shapely", "geopandas", "ultralytics"):
        try:
            m = __import__(mod)
            print(f"{mod}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"[缺失] {mod}")


if __name__ == "__main__":
    main()
