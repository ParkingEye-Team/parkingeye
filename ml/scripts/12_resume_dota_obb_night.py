"""夜间 DOTA-OBB 续训（低显存稳妥配置）。

崩溃根因: make_anchors 阶段偶发 CUDA illegal memory access，由大实例 batch 拉高显存峰值。
规避: batch 2 + imgsz 832 + workers 0（Windows DataLoader 崩 OpenCV）。

用法:
    ./.venv/Scripts/python.exe ml/scripts/12_resume_dota_obb_night.py
"""
from ultralytics import YOLO

model = YOLO("runs/obb/dota_vehicle_obb/weights/last.pt")  # epoch 43 的 last.pt

model.train(
    resume=True,          # 从 runs/obb/dota_vehicle_obb 读 args.yaml + last.pt 续跑
    batch=2,              # 8GB 显存, 降峰值
    imgsz=832,            # 原 896 -> 832, 省 ~15% 激活显存
    workers=0,            # Windows DataLoader 规避
    device=0,
)
