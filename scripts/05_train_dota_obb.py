"""DOTA 车辆检测预训练（YOLO-OBB）。

在 data/dota_yolo/dota_obb.yaml 上训练旋转框车辆检测器。
RTX 5060 8GB 配置：YOLO11n/s-OBB, imgsz 1024, batch 8, FP16。

用法:
  python scripts/05_train_dota_obb.py            # 全量训练
  python scripts/05_train_dota_obb.py --epochs 60 --batch 8
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=Path("data/dota_yolo/dota_obb.yaml"))
    ap.add_argument("--model", default="yolo11n-obb.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)   # 8GB 显存安全值
    ap.add_argument("--imgsz", type=int, default=896)  # 略小于1024，降显存
    ap.add_argument("--device", default=0)
    ap.add_argument("--name", default="dota_vehicle_obb")
    args = ap.parse_args()

    if not args.yaml.exists():
        raise SystemExit(f"数据 yaml 不存在: {args.yaml}，先运行 04 脚本转换")

    model = YOLO(args.model)
    model.train(
        data=str(args.yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        exist_ok=True,       # OBB 任务默认输出到 runs/obb/<name>
        amp=True,            # FP16
        patience=15,
        workers=4,
        cache=False,         # 8GB 显存不缓存
    )


if __name__ == "__main__":
    main()
