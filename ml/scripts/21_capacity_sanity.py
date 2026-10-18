"""容量一致性自检：估计容量 vs 影像中检测到的车辆数。

不依赖人工真值，回答两个问题：
  1. 是否存在"检测车辆数 > 估计容量"的场地（说明容量低估或检测误报）；
  2. 各场地"检测车数 / 估计容量"的比例分布（只能作为占用率下界，因为检测漏检严重）。

方法：
  - 对容量 AOI 做 DOTA 检测滑窗（imgsz1024 / max_det2000 / conf0.25，RGB 约定见 P0-3）；
  - 车辆 OBB 中心 -> UTM；
  - 跨窗口按中心距离去重（默认 2m），同时输出 raw 与 dedup 两套计数；
  - 逐个容量多边形统计内部车辆数；跳过 excluded=True 或 exclude_lots.csv 中的地块；
  - 输出 data/capacity/capacity_sanity.csv（覆盖前自动备份）。

口径：
  - raw 是滑窗检出之和，会重复计数；dedup 才是接近“唯一车辆数”的量级参考；
  - 检测召回受 0.5m 分辨率限制，比例只能当下界，不能当真实占用率。

用法:
  ./.venv/Scripts/python.exe ml/scripts/21_capacity_sanity.py
  ./.venv/Scripts/python.exe ml/scripts/21_capacity_sanity.py --dedup 5 --out data/capacity/capacity_sanity_5m.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from shapely.geometry import shape, Point
from ultralytics import YOLO

from safe_io import run_id, write_csv_safe

ROOT = Path(__file__).resolve().parents[2]
TIF = ROOT / "data/imagery/JL1KF01C/JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH" \
      / "JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH.tif"
DET_W = ROOT / "runs/obb/dota_vehicle_obb/weights/best.pt"
CAP_DIR = ROOT / "data/capacity"
CAP_GJ = CAP_DIR / "aoi_capacity_utm.geojson"
META = CAP_DIR / "aoi_capacity_meta.json"
OUT = CAP_DIR / "capacity_sanity.csv"
CONF, MAX_DET = 0.25, 2000


def tile_positions(size, tile, stride):
    if size <= tile:
        return [0]
    pos = list(range(0, size - tile, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def dedup_centers(points, min_dist):
    """按欧氏距离去重（网格加速），保留先出现者。"""
    if min_dist <= 0:
        return list(points)
    cell = float(min_dist)
    grid = {}
    kept = []
    for x, y in points:
        ix, iy = int(x // cell), int(y // cell)
        dup = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in grid.get((ix + dx, iy + dy), ()):
                    if (x - px) ** 2 + (y - py) ** 2 < min_dist ** 2:
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            kept.append((x, y))
            grid.setdefault((ix, iy), []).append((x, y))
    return kept


def load_excluded():
    """合并 exclude_lots.csv 与 GeoJSON 的 excluded=True。"""
    ids = set()
    p = CAP_DIR / "exclude_lots.csv"
    if p.exists():
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("lot_id"):
                    ids.add(r["lot_id"].strip())
    return ids


def main():
    ap = argparse.ArgumentParser(description="容量一致性自检")
    ap.add_argument("--col", type=int, default=None)
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--dedup", type=float, default=2.0, help="跨窗口去重距离（米），0 表示不去重")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    col0 = args.col if args.col is not None else meta.get("aoi", [1603, 15201, 4096])[0]
    row0 = args.row if args.row is not None else meta.get("aoi", [1603, 15201, 4096])[1]
    size = args.size if args.size is not None else meta.get("aoi", [1603, 15201, 4096])[2]
    rid = run_id()
    print(f"run_id={rid} | AOI=({col0},{row0}) {size}x{size} | tile={args.tile} | "
          f"dedup={args.dedup}m")

    det = YOLO(str(DET_W))
    raw_centers = []  # UTM
    with rasterio.open(TIF) as src:
        win = rasterio.windows.Window(col0, row0, size, size)
        wt = src.window_transform(win)
        arr = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0)).astype(np.uint8)
        pos = tile_positions(size, args.tile, args.stride)
        n_win = 0
        for y0 in pos:
            for x0 in pos:
                tile = arr[y0:y0 + args.tile, x0:x0 + args.tile]
                if tile.shape[0] != args.tile or tile.shape[1] != args.tile:
                    continue
                # DOTA 旧权重保持 RGB（P0-3）
                r = det(tile, conf=CONF, iou=0.55, imgsz=args.tile,
                        max_det=MAX_DET, verbose=False)[0]
                n_win += 1
                if r.obb is None:
                    continue
                for p in r.obb.xyxyxyxy.cpu().numpy():
                    cx, cy = p.mean(axis=0)
                    gx = wt * (x0 + cx, y0 + cy)
                    raw_centers.append((gx[0], gx[1]))
    dedup = dedup_centers(raw_centers, args.dedup)
    print(f"检测完成: {n_win} 窗口 | raw {len(raw_centers)} | dedup {len(dedup)}（{args.dedup}m）")

    gj = json.loads(CAP_GJ.read_text(encoding="utf-8"))
    exclude_ids = load_excluded()
    rows = []
    skipped = []
    for f in gj["features"]:
        p = f["properties"]
        if p.get("excluded") or p.get("id") in exclude_ids:
            skipped.append(p.get("id"))
            continue
        g = shape(f["geometry"])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        n_raw = sum(1 for x, y in raw_centers if g.contains(Point(x, y)))
        n_dedup = sum(1 for x, y in dedup if g.contains(Point(x, y)))
        cap = int(p["capacity"])
        rows.append({
            "lot_id": p["id"], "area_m2": p["area_m2"], "capacity": cap,
            "detected_raw": n_raw,
            "detected_dedup": n_dedup,
            "dedup_over_capacity": round(n_dedup / cap, 3) if cap else "",
            "raw_over_capacity": round(n_raw / cap, 3) if cap else "",
            "flag": "CHECK" if n_dedup > cap else "",
        })
    rows.sort(key=lambda r: -r["detected_dedup"])

    out = Path(args.out)
    fields = ["lot_id", "area_m2", "capacity", "detected_raw", "detected_dedup",
              "dedup_over_capacity", "raw_over_capacity", "flag"]
    write_csv_safe(out, rows, fieldnames=fields, encoding="utf-8", tag="sanity")
    if skipped:
        print(f"排除地块 {len(skipped)} 个: {skipped}")

    flags = [r for r in rows if r["flag"] == "CHECK"]
    total_raw = sum(r["detected_raw"] for r in rows)
    total_dedup = sum(r["detected_dedup"] for r in rows)
    total_cap = sum(r["capacity"] for r in rows)
    print(f"逐场地结果 -> {out}")
    if total_cap:
        print(f"  检测车辆（dedup）{total_dedup} / 估计容量合计 {total_cap} = "
              f"{total_dedup / total_cap * 100:.1f}%（漏检严重，仅作下界）")
        print(f"  raw 窗口检出之和 {total_raw}（会重复计数，不作唯一车辆数）")
    print(f"  检测数 > 容量的场地: {len(flags)} 个 {[r['lot_id'] for r in flags]}")
    print("  前5个场地:")
    for r in rows[:5]:
        print(f"    {r['lot_id']}: 容量 {r['capacity']}, dedup {r['detected_dedup']} "
              f"({r['dedup_over_capacity']}), raw {r['detected_raw']}")


if __name__ == "__main__":
    main()
