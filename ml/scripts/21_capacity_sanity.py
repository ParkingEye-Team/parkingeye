"""容量一致性自检：估计容量 vs 影像中检测到的车辆数。

不依赖人工真值，回答两个问题：
  1. 是否存在"检测车辆数 > 估计容量"的场地（说明容量低估或检测误报）；
  2. 各场地"检测车数 / 估计容量"的比例分布（只能作为占用率下界，因为检测漏检严重）。

方法：
  - 对容量 AOI 做 DOTA 检测滑窗（imgsz1024 / max_det2000 / conf0.25，RGB 约定见 P0-3）；
  - 车辆 OBB 中心 -> UTM；
  - 逐个容量多边形统计内部车辆数（可用面积法生成的数据）；
  - 输出 data/capacity/capacity_sanity.csv 与摘要。

用法:
  ./.venv/Scripts/python.exe ml/scripts/21_capacity_sanity.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import transform as wtransform
from shapely.geometry import shape, Point
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
TIF = ROOT / "data/imagery/JL1KF01C/JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH" \
      / "JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH.tif"
DET_W = ROOT / "runs/obb/dota_vehicle_obb/weights/best.pt"
CAP_GJ = ROOT / "data/capacity/aoi_capacity_utm.geojson"
OUT = ROOT / "data/capacity/capacity_sanity.csv"
TILE, STRIDE = 1024, 768
CONF, MAX_DET = 0.25, 2000


def tile_positions(size, tile, stride):
    if size <= tile:
        return [0]
    pos = list(range(0, size - tile, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def main():
    # AOI 与 18 号脚本一致：col=1603,row=15201,size=4096
    col0, row0, size = 1603, 15201, 4096
    det = YOLO(str(DET_W))
    centers = []  # UTM
    with rasterio.open(TIF) as src:
        win = rasterio.windows.Window(col0, row0, size, size)
        wt = src.window_transform(win)
        arr = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0)).astype(np.uint8)
        pos = tile_positions(size, TILE, STRIDE)
        n_win = 0
        for y0 in pos:
            for x0 in pos:
                tile = arr[y0:y0 + TILE, x0:x0 + TILE]
                if tile.shape[0] != TILE or tile.shape[1] != TILE:
                    continue
                # DOTA 旧权重保持 RGB（P0-3）
                r = det(tile, conf=CONF, iou=0.55, imgsz=TILE, max_det=MAX_DET, verbose=False)[0]
                n_win += 1
                if r.obb is None:
                    continue
                for p in r.obb.xyxyxyxy.cpu().numpy():
                    cx, cy = p.mean(axis=0)
                    # 窗口内像素 -> AOI 像素 -> UTM
                    gx = wt * (x0 + cx, y0 + cy)
                    centers.append((gx[0], gx[1]))
    print(f"检测完成: {n_win} 窗口, {len(centers)} 辆车")

    gj = json.load(open(CAP_GJ, encoding="utf-8"))
    rows = []
    for f in gj["features"]:
        p = f["properties"]
        g = shape(f["geometry"])
        # 只统计主多边形（MultiPolygon 取最大块）
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        n = sum(1 for x, y in centers if g.contains(Point(x, y)))
        cap = int(p["capacity"])
        rows.append({
            "lot_id": p["id"], "area_m2": p["area_m2"], "capacity": cap,
            "detected_vehicles": n,
            "detected_over_capacity": round(n / cap, 3) if cap else "",
            "flag": "CHECK" if n > cap else "",
        })
    rows.sort(key=lambda r: -(r["detected_vehicles"]))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    flags = [r for r in rows if r["flag"] == "CHECK"]
    total_det = sum(r["detected_vehicles"] for r in rows)
    total_cap = sum(r["capacity"] for r in rows)
    print(f"逐场地结果 -> {OUT}")
    print(f"  检测车辆总数 {total_det} / 估计容量合计 {total_cap} = {total_det/total_cap*100:.1f}%（漏检严重，仅作下界）")
    print(f"  检测数 > 容量的场地: {len(flags)} 个 {[r['lot_id'] for r in flags]}")
    print("  前5个场地:")
    for r in rows[:5]:
        print(f"    {r['lot_id']}: 容量 {r['capacity']}, 检出 {r['detected_vehicles']} ({r['detected_over_capacity']})")


if __name__ == "__main__":
    main()