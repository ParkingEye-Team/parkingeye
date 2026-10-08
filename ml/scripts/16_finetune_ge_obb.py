"""GE 成都域微调 DOTA 车辆检测权重（目标：提升成都域召回）。

输入：data/ge_yolo_tiled/train（1024 切片，来自 4 张 GE 大图 / 462 原始框 / 554 切片框）
      runs/obb/dota_vehicle_obb/weights/best.pt（DOTA 预训练）
输出：runs/obb/ge_finetune/

说明：
- 切片 1024，imgsz 1024，batch 2（高车辆密度切片显存峰值可控），workers 0（Windows）。
- val 与 train 同源（仅 4 张大图，不再切 val 削弱训练）；训练内 val 指标只作 loss 监控，
  微调效果以独立评估为准（冬季吉林一号窗口 + 未标注 GE 图目视）。
- 数据量小（37 切片），epochs 40 + patience 15 防过拟合；DOTA 权重作初始化做全量微调。
"""
from ultralytics import YOLO
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# yaml: 指向切片集；val 复用 train（见文件头说明）
yaml_path = ROOT / "data/ge_yolo_tiled" / "ge_obb_tiled.yaml"
yaml_path.write_text(yaml.safe_dump({
    "path": str(ROOT / "data/ge_yolo_tiled"),
    "train": "train/images",
    "val": "train/images",
    "names": {0: "vehicle"},
}, allow_unicode=True, sort_keys=False), encoding="utf-8")

model = YOLO(str(ROOT / "runs/obb/dota_vehicle_obb/weights/best.pt"))
model.train(
    data=str(yaml_path),
    epochs=40,
    imgsz=1024,
    batch=2,
    workers=0,
    device=0,
    patience=15,
    close_mosaic=10,
    name="ge_finetune",
    exist_ok=True,
)
