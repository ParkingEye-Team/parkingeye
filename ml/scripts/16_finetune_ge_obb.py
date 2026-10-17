"""GE 成都域微调 DOTA 车辆检测权重（目标：提升成都域召回）。

输入：data/ge_yolo_tiled/{train,val}（1024 切片，来自 12 张 GE 大图 / 2139 原始框）
      train: 10 张大图 -> 104 切片 / 2178 框
      val:   2 张大图（真留出，不参与训练）-> 17 切片 / 234 框
      runs/obb/dota_vehicle_obb/weights/best.pt（DOTA 预训练）
输出：runs/obb/ge_finetune_v2/

说明：
- 切片 1024，imgsz 1024，batch 2，workers 0（Windows）。
- v2 使用真留出 val（v1 曾 val=train 同源，指标虚高不可比）。
- 评估还要看独立域：GE 留出图 + 夏季/冬季吉林一号窗口（见评估脚本）。
"""
from ultralytics import YOLO
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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
    name="ge_finetune_v2",
    exist_ok=True,
)
