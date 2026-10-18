"""GE 成都域微调 DOTA 车辆检测权重（目标：提升 GE 域检测）。

输入：data/ge_yolo_tiled/{train,val}（1024 切片，含边缘补窗与少量背景切片）
      runs/obb/dota_vehicle_obb/weights/best.pt（DOTA 预训练）
输出：runs/obb/ge_finetune_v3/

说明：
- v3 使用修正后的切片集（P0-5 边缘补窗 + P1-1 背景切片），val 为真留出图。
- 切片 1024，imgsz 1024，batch 2，workers 0（Windows）。
- 评估还要看独立域：GE 留出图 + 夏季/冬季吉林一号窗口（见 19 号脚本）。
- 注意：DOTA 旧权重存在 R/B 交换的历史问题（doc/records/audit-2026-10-17.md P0-3），本脚本不改通道约定，
  仅加载权重继续训练；推理端保持现有 RGB 约定。

用法:
  ./.venv/Scripts/python.exe ml/scripts/16_finetune_ge_obb.py
"""
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def main():
    # yaml: 指向切片集，val 为真留出
    yaml_path = ROOT / "data/ge_yolo_tiled" / "ge_obb_tiled.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(ROOT / "data/ge_yolo_tiled"),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "vehicle"},
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")

    model = YOLO(str(ROOT / "runs/obb/dota_vehicle_obb/weights/best.pt"))
    model.train(
        data=str(yaml_path),
        epochs=60,
        imgsz=1024,
        batch=2,
        workers=0,
        device=0,
        patience=20,
        close_mosaic=10,
        name="ge_finetune_v3",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()