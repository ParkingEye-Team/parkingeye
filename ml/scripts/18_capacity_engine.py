"""容量估计引擎 MVP：停车场多边形 -> 面积 -> 布局系数 -> 容量估计。

数据流：
  冬季影像 AOI 窗口
    ├─ 分割模型滑窗推理 -> 停车场掩膜 -> 矢量化为多边形（UTM 精确面积）
    └─ OSM 停车场多边形（AOI 内）-> 投影到 UTM
  对每个多边形：capacity = area / coef(layout)，并给出 ±20% 区间

输出（data/capacity/）：
  - aoi_capacity_utm.geojson   多边形（UTM，面积精确）
  - aoi_capacity_wgs84.geojson 多边形（WGS84，供 Web 用）
  - aoi_capacity.csv           逐场地表
  - aoi_capacity_preview.png   容量专题图
  - aoi_capacity_meta.json     运行元数据（权重、参数、口径说明）

用法:
  ./.venv/Scripts/python.exe ml/scripts/18_capacity_engine.py [--col 1603 --row 15201 --size 4096]
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.windows
from PIL import Image, ImageDraw, ImageFont
from rasterio import features
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping, Polygon
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
TIF = ROOT / "data/imagery/JL1KF01C/JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH" \
      / "JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH.tif"
SEG_W = ROOT / "runs/segment/parking_seg_night/weights/best.pt"
OSM = ROOT / "data/osm/chengdu_parking_raw.json"
OUT = ROOT / "data/capacity"
UTM = "EPSG:32648"
WGS = "EPSG:4326"

# 布局单车位综合占地面积（m²/车位，含通道摊销）——MVP 默认值，待真值校准（doc/02 第4周）
COEF = {"垂直式": 28, "斜列式": 24, "平行式": 36, "不规则": 32}
DEFAULT_LAYOUT = "垂直式"
MIN_AREA = 200.0        # m²，过滤小噪点
TILE, STRIDE = 1024, 768
CONF = 0.25
COEF_UNCERTAINTY = 0.20  # 系数 ±20% -> 容量区间


def tile_positions(size, tile, stride):
    if size <= tile:
        return [0]
    pos = list(range(0, size - tile, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def load_font(sz=18):
    for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", type=int, default=1603)
    ap.add_argument("--row", type=int, default=15201)
    ap.add_argument("--size", type=int, default=4096)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) 读 AOI 影像
    with rasterio.open(TIF) as src:
        win = rasterio.windows.Window(args.col, args.row, args.size, args.size)
        wt = src.window_transform(win)
        arr = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0)).astype(np.uint8)  # RGB
        bounds = rasterio.transform.array_bounds(args.size, args.size, wt)  # (left, bottom, right, top)
    print(f"AOI: ({args.col},{args.row}) {args.size}x{args.size}, 影像 {arr.shape[1]}x{arr.shape[0]}")

    # 2) 分割滑窗推理 -> 掩膜
    seg = YOLO(str(SEG_W))
    mask = np.zeros(arr.shape[:2], np.uint8)
    poss = tile_positions(args.size, TILE, STRIDE)
    n_tiles = 0
    for y0 in poss:
        for x0 in poss:
            tile = arr[y0:y0 + TILE, x0:x0 + TILE]
            bgr = np.ascontiguousarray(tile[:, :, ::-1])
            r = seg(bgr, imgsz=TILE, conf=CONF, verbose=False)[0]
            n_tiles += 1
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy():
                    mm = cv2.resize((m * 255).astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
                    region = mask[y0:y0 + TILE, x0:x0 + TILE]
                    mask[y0:y0 + TILE, x0:x0 + TILE] = np.maximum(region, mm)
    print(f"分割推理: {n_tiles} 切片, 掩膜覆盖 {mask.mean()/255*100:.1f}%")

    # 3) 矢量化（UTM 坐标，面积即 m²）
    items = []  # (source, shapely polygon)
    for geom, _val in features.shapes(mask, mask=mask > 0, transform=wt):
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        for p in parts:
            if p.area >= MIN_AREA:
                items.append(("seg", p))
    print(f"分割矢量: {len(items)} 个多边形(>= {MIN_AREA:.0f} m²)")

    # 4) OSM 多边形（AOI 内）
    osm_n = 0
    els = json.load(open(OSM, encoding="utf-8"))["elements"]
    for e in els:
        if "geometry" not in e:
            continue
        coords = [(g["lon"], g["lat"]) for g in e["geometry"]]
        if len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            g_utm = shape(transform_geom(WGS, UTM, mapping(poly)))
        except Exception:
            continue
        rp = g_utm.representative_point()
        if bounds[0] <= rp.x <= bounds[2] and bounds[1] <= rp.y <= bounds[3] and g_utm.area >= MIN_AREA:
            items.append(("osm", g_utm))
            osm_n += 1
    print(f"OSM 多边形(AOI 内): {osm_n} 个")

    # 5) 容量计算
    coef = COEF[DEFAULT_LAYOUT]
    rows = []
    for i, (source, g) in enumerate(items):
        a = g.area
        rows.append({
            "id": f"{source}_{i:03d}", "source": source,
            "area_m2": round(a, 1), "layout": DEFAULT_LAYOUT, "coef_m2": coef,
            "capacity": round(a / coef),
            "cap_min": round(a / (coef * (1 + COEF_UNCERTAINTY))),
            "cap_max": round(a / (coef * (1 - COEF_UNCERTAINTY))),
            "cx": round(g.representative_point().x, 1), "cy": round(g.representative_point().y, 1),
        })
    tot_cap = sum(r["capacity"] for r in rows)
    seg_rows = [r for r in rows if r["source"] == "seg"]
    osm_rows = [r for r in rows if r["source"] == "osm"]
    print(f"容量: 总 {tot_cap} 车位（seg {sum(r['capacity'] for r in seg_rows)} / osm {sum(r['capacity'] for r in osm_rows)}）")

    # 6) 输出 GeoJSON / CSV / 预览图 / meta
    feats_utm = [{"type": "Feature", "geometry": mapping(g), "properties": r}
                 for (source, g), r in zip(items, rows)]
    (OUT / "aoi_capacity_utm.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": UTM}}, "features": feats_utm},
                   ensure_ascii=False), encoding="utf-8")
    feats_wgs = [{"type": "Feature", "geometry": transform_geom(UTM, WGS, mapping(g)), "properties": r}
                 for (source, g), r in zip(items, rows)]
    (OUT / "aoi_capacity_wgs84.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats_wgs}, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "aoi_capacity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 预览图
    img = Image.fromarray(arr)
    dr = ImageDraw.Draw(img)
    font = load_font(20)
    for (source, g), r in zip(items, rows):
        color = (0, 230, 110) if source == "seg" else (255, 150, 0)
        xy = [(float(x), float(y)) for x, y in g.exterior.coords]
        # UTM -> 像素
        px = [(int((x - wt.c) / wt.a), int((y - wt.f) / wt.e)) for x, y in xy]
        dr.line(px, fill=color, width=2)
        dr.text((px[0][0] + 3, px[0][1] + 3), f"{r['capacity']}", fill=color, font=font)
    img.save(OUT / "aoi_capacity_preview.png")

    (OUT / "aoi_capacity_meta.json").write_text(json.dumps({
        "tif": str(TIF), "aoi": [args.col, args.row, args.size], "seg_weights": str(SEG_W),
        "conf": CONF, "min_area_m2": MIN_AREA, "coef_table": COEF, "default_layout": DEFAULT_LAYOUT,
        "coef_uncertainty": COEF_UNCERTAINTY,
        "counts": {"total": len(rows), "seg": len(seg_rows), "osm": len(osm_rows)},
        "capacity_total": tot_cap,
        "caveat": "布局为默认值(垂直式)，系数为工程初值，待容量真值校准；AOI 与标注训练区有重叠，分割表现为 in-sample 偏乐观",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出 -> {OUT}")
    print("  aoi_capacity_preview.png / .csv / _utm.geojson / _wgs84.geojson / _meta.json")


if __name__ == "__main__":
    main()
