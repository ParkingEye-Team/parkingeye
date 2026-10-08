"""夏季影像正式评估：分割泛化 + 车辆检测（修正配置）。

对夏季景(JL1KF01C 2024-08-25, 成都东部)内所有 OSM 停车场窗口：
  - 分割模型(冬季训练) -> 测试跨域泛化：能否圈出停车场
  - 车辆检测(修正配置 imgsz=1024/max_det=2000) -> 检出数
  - 保存叠加图供人工复核
输出：data/infer/summer_eval/ + CSV

用法:
  ./.venv/Scripts/python.exe ml/scripts/17_eval_summer.py
"""
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import transform as wtransform
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
TIF = ROOT / "data/imagery/JL1KF01C_PMSL1_20240825114907_200295580_102_0009_001_L3D_PSH" \
      / "JL1KF01C_PMSL1_20240825114907_200295580_102_0009_001_L3D_PSH.tif"
DET_W = ROOT / "runs/obb/dota_vehicle_obb/weights/best.pt"
SEG_W = ROOT / "runs/segment/parking_seg_night/weights/best.pt"
OSM = ROOT / "data/osm/chengdu_parking_raw.json"
OUT = ROOT / "data/infer/summer_eval"

IMGSZ = 1024
MAX_DET = 2000
CONF = 0.25


def main():
    import json
    OUT.mkdir(parents=True, exist_ok=True)
    det = YOLO(str(DET_W))
    seg = YOLO(str(SEG_W))
    els = [e for e in json.load(open(OSM, encoding="utf-8"))["elements"] if "geometry" in e]
    lons = [np.mean([g["lon"] for g in e["geometry"]]) for e in els]
    lats = [np.mean([g["lat"] for g in e["geometry"]]) for e in els]

    rows = []
    with rasterio.open(TIF) as src:
        H, W = src.height, src.width
        xs, ys = wtransform("EPSG:4326", src.crs, lons, lats)
        checked = 0
        for i, (x, y) in enumerate(zip(xs, ys)):
            col, row = ~src.transform * (x, y)
            col, row = int(col), int(row)
            if not (0 <= col < W and 0 <= row < H):
                continue
            c0 = max(0, min(col - 512, W - IMGSZ))
            r0 = max(0, min(row - 512, H - IMGSZ))
            arr = np.transpose(src.read([1, 2, 3], window=rasterio.windows.Window(c0, r0, IMGSZ, IMGSZ)), (1, 2, 0)).astype(np.uint8)
            if (arr.max(axis=2) < 8).mean() > 0.5:  # 死区
                continue
            checked += 1
            # 分割
            rs = seg(arr, imgsz=IMGSZ, conf=0.25, verbose=False)[0]
            n_seg = 0 if rs.masks is None else len(rs.masks)
            # 检测
            rd = det(arr, conf=CONF, iou=0.55, imgsz=IMGSZ, max_det=MAX_DET, verbose=False)[0]
            n_det = 0 if rd.obb is None else len(rd.obb)
            rows.append({"idx": i, "col": c0, "row": r0, "seg_lots": n_seg, "vehicles": n_det})
            vis = arr.copy()
            if n_det:
                for p in rd.obb.xyxyxyxy.cpu().numpy():
                    cv2.polylines(vis, [p.astype(np.int32)], True, (0, 255, 0), 1)
            if rs.masks is not None:
                for m_ in rs.masks.data.cpu().numpy():
                    mm = cv2.resize((m_ * 255).astype(np.uint8), (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)
                    cont, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cont, -1, (255, 0, 0), 1)
            cv2.imwrite(str(OUT / f"osm{i:03d}_seg{n_seg}_det{n_det}.jpg"), vis[:, :, ::-1])

    if rows:
        import numpy as np2
        ns = [r["vehicles"] for r in rows]
        ss = [r["seg_lots"] for r in rows]
        print(f"夏季景评估: {len(rows)} 个停车场窗口")
        print(f"  分割: 检出停车场的窗口 {(sum(1 for s in ss if s>0))}, 共 {sum(ss)} 个实例")
        print(f"  检测: 有检出窗口 {sum(1 for n in ns if n>0)}, 总 {sum(ns)} 辆, 中位 {np2.median(ns):.0f}, 最大 {max(ns)}")
        print(f"  零检出窗口: {sum(1 for n in ns if n==0)}")
        with open(OUT / "summer_eval.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"结果: {OUT}")
    else:
        print("无有效窗口")


if __name__ == "__main__":
    main()
