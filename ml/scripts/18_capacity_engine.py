"""容量估计引擎：停车场多边形 -> 面积 -> 布局系数 -> 容量估计。

数据流：
  冬季影像 AOI 窗口
    ├─ 分割模型滑窗推理 -> 停车场掩膜 -> 矢量化为多边形（UTM 精确面积，输入 BGR）
    └─ OSM 停车场多边形（AOI 内）-> 投影到 UTM
  两来源按几何重叠匹配去重（同一物理地块只计一次，几何取并集）
  对每个唯一地块：capacity = union_area / coef(layout)
  区间覆盖布局不确定性（平行 36 / 垂直 28 / 斜列 24 m²每车位），不是统计置信区间

输出（data/capacity/）：
  - aoi_capacity_utm.geojson   唯一地块（UTM，面积精确）
  - aoi_capacity_wgs84.geojson 唯一地块（WGS84，供 Web 用）
  - aoi_capacity.csv           逐场地表（含 source/matched_id/多布局容量）
  - aoi_capacity_preview.png   容量专题图
  - aoi_capacity_meta.json     运行元数据（权重、参数、口径说明）
  - truth_collection_sheet.csv 容量真值采集工单（唯一地块）

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
from rasterio.warp import transform_geom, transform as wtransform
from shapely.geometry import shape, mapping, Polygon
from ultralytics import YOLO
from osm_utils import is_surface_parking

ROOT = Path(__file__).resolve().parents[2]
TIF = ROOT / "data/imagery/JL1KF01C/JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH" \
      / "JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH.tif"
SEG_W = ROOT / "runs/segment/parking_seg_night/weights/best.pt"
OSM = ROOT / "data/osm/chengdu_parking_raw.json"
OUT = ROOT / "data/capacity"
UTM = "EPSG:32648"
WGS = "EPSG:4326"

# 布局单车位综合占地面积（m²/车位，含通道摊销）——工程初值，待真值校准
COEF = {"垂直式": 28, "斜列式": 24, "平行式": 36, "不规则": 32}
DEFAULT_LAYOUT = "垂直式"
MIN_AREA = 200.0          # m²，过滤小噪点
TILE, STRIDE = 1024, 768
CONF = 0.25
MATCH_MIN_RATIO = 0.50    # 跨来源重叠占较小者比例 >= 该值 -> 视为同一地块（包含关系优先）
MATCH_MIN_IOU = 0.15      # IoU 门槛放宽：OSM 与分割边界大小不一，IoU 会天然偏低


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


def safe_area(g):
    try:
        return g.intersection(g).area
    except Exception:
        return 0.0


def merge_lots(seg_polys, osm_polys, min_ratio=MATCH_MIN_RATIO, min_iou=MATCH_MIN_IOU):
    """跨来源匹配去重：同一物理地块只保留一条，几何取并集。

    返回 list[dict]: id, source(seg/osm/both), matched_id, geom, area_seg, area_osm
    """
    used_seg = set()
    lots = []
    for j, (osm_id, g_osm) in enumerate(osm_polys):
        best_i, best_ratio, best_iou = None, 0.0, 0.0
        for i, g_seg in enumerate(seg_polys):
            if i in used_seg:
                continue
            try:
                inter = g_osm.intersection(g_seg).area
            except Exception:
                continue
            if inter <= 0:
                continue
            ratio = inter / min(g_osm.area, g_seg.area)
            iou = inter / (g_osm.area + g_seg.area - inter)
            if ratio > best_ratio:
                best_i, best_ratio, best_iou = i, ratio, iou
        if best_i is not None and best_ratio >= min_ratio and best_iou >= min_iou:
            g_seg = seg_polys[best_i]
            used_seg.add(best_i)
            try:
                merged = g_osm.union(g_seg)
            except Exception:
                merged = g_osm
            lots.append({
                "id": f"seg_{best_i:03d}", "source": "both",
                "matched_id": f"osm_{j:02d}:{osm_id}",
                "geom": merged, "area_seg": g_seg.area, "area_osm": g_osm.area,
            })
        else:
            lots.append({
                "id": f"osm_{j:02d}", "source": "osm", "matched_id": "",
                "geom": g_osm, "area_seg": 0.0, "area_osm": g_osm.area,
            })
    for i, g in enumerate(seg_polys):
        if i not in used_seg:
            lots.append({
                "id": f"seg_{i:03d}", "source": "seg", "matched_id": "",
                "geom": g, "area_seg": g.area, "area_osm": 0.0,
            })
    return lots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", type=int, default=1603)
    ap.add_argument("--row", type=int, default=15201)
    ap.add_argument("--size", type=int, default=4096)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) 读 AOI 影像（先钳制边界，P2-2）
    with rasterio.open(TIF) as src:
        W_img, H_img = src.width, src.height
        size = min(args.size, W_img, H_img)
        col = max(0, min(args.col, W_img - size))
        row = max(0, min(args.row, H_img - size))
        win = rasterio.windows.Window(col, row, size, size)
        wt = src.window_transform(win)
        arr = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0)).astype(np.uint8)  # RGB
        bounds = rasterio.transform.array_bounds(size, size, wt)  # (left, bottom, right, top)
    print(f"AOI: ({col},{row}) {size}x{size}, 影像 {arr.shape[1]}x{arr.shape[0]}")

    # 2) 分割滑窗推理 -> 掩膜（BGR，训练约定；P0-2）
    seg = YOLO(str(SEG_W))
    mask = np.zeros((size, size), np.uint8)
    poss = tile_positions(size, TILE, STRIDE)
    n_tiles = 0
    for y0 in poss:
        for x0 in poss:
            tile = arr[y0:y0 + TILE, x0:x0 + TILE]
            if tile.shape[0] != TILE or tile.shape[1] != TILE:
                continue
            bgr = np.ascontiguousarray(tile[:, :, ::-1])
            r = seg(bgr, imgsz=TILE, conf=CONF, verbose=False)[0]
            n_tiles += 1
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy():
                    mm = cv2.resize((m * 255).astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
                    region = mask[y0:y0 + TILE, x0:x0 + TILE]
                    mask[y0:y0 + TILE, x0:x0 + TILE] = np.maximum(region, mm)
    print(f"分割推理: {n_tiles} 切片, 掩膜覆盖 {mask.mean()/255*100:.1f}%")

    # 3) 分割矢量（UTM）
    seg_polys = []
    for geom, _val in features.shapes(mask, mask=mask > 0, transform=wt):
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        for p in parts:
            if p.area >= MIN_AREA:
                seg_polys.append(p)
    print(f"分割矢量: {len(seg_polys)} 个多边形(>= {MIN_AREA:.0f} m²)")

    # 4) OSM 多边形（AOI 内）
    osm_polys = []
    els = json.load(open(OSM, encoding="utf-8"))["elements"]
    for e in els:
        if "geometry" not in e:
            continue
        if not is_surface_parking(e.get("tags", {})):
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
            osm_polys.append((e.get("id"), g_utm))
    print(f"OSM 多边形(AOI 内): {len(osm_polys)} 个")

    # 5) 跨来源匹配去重（P0-1）：同一物理地块只保留一条，几何取并集
    lots = merge_lots(seg_polys, osm_polys, MATCH_MIN_RATIO, MATCH_MIN_IOU)
    matched = sum(1 for x in lots if x["source"] == "both")
    print(f"匹配去重: 唯一地块 {len(lots)} 个（其中双来源匹配 {matched} 对，已合并）")

    # 6) 容量计算（多布局区间，P1-5）
    # 可选人工排除清单：data/capacity/exclude_lots.csv（lot_id,reason）
    exclude_ids = set()
    exclude_csv = OUT / "exclude_lots.csv"
    if exclude_csv.exists():
        with open(exclude_csv, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("lot_id"):
                    exclude_ids.add(r["lot_id"].strip())
        if exclude_ids:
            print(f"人工排除清单: {len(exclude_ids)} 个地块 {sorted(exclude_ids)}")

    coef_v = COEF["垂直式"]; coef_d = COEF["斜列式"]; coef_p = COEF["平行式"]
    rows = []
    for lot in lots:
        a = lot["geom"].area
        rp = lot["geom"].representative_point()
        rows.append({
            "id": lot["id"], "source": lot["source"], "matched_id": lot["matched_id"],
            "area_m2": round(a, 1),
            "area_seg_m2": round(lot["area_seg"], 1), "area_osm_m2": round(lot["area_osm"], 1),
            "layout": DEFAULT_LAYOUT, "coef_m2": coef_v,
            "capacity": round(a / coef_v),          # 默认按垂直式
            "cap_diag": round(a / coef_d),          # 斜列 24
            "cap_para": round(a / coef_p),          # 平行 36
            "cap_min": round(a / coef_p),           # 区间下界 = 平行布局
            "cap_max": round(a / coef_d),           # 区间上界 = 斜列布局
            "excluded": lot["id"] in exclude_ids,   # 人工判定为非停车场/不可计入
            "cx": round(rp.x, 1), "cy": round(rp.y, 1),
        })
    if not rows:  # P2-1 空结果保护
        print("警告：AOI 内没有有效停车场多边形，写出空产物")
        with open(OUT / "aoi_capacity.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id", "source", "matched_id", "area_m2", "area_seg_m2",
                                    "area_osm_m2", "layout", "coef_m2", "capacity",
                                    "cap_diag", "cap_para", "cap_min", "cap_max", "excluded", "cx", "cy"])
        (OUT / "aoi_capacity_utm.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False), encoding="utf-8")
        (OUT / "aoi_capacity_wgs84.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False), encoding="utf-8")
        (OUT / "truth_collection_sheet.csv").write_text("lot_id,lon,lat,area_m2,model_capacity,true_capacity,layout_true,notes\n", encoding="utf-8-sig")
        (OUT / "aoi_capacity_meta.json").write_text(json.dumps({"counts": {"total": 0}, "capacity_total_unique": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    active = [r for r in rows if not r["excluded"]]
    excluded = [r for r in rows if r["excluded"]]
    tot_cap = sum(r["capacity"] for r in active)
    tot_min = sum(r["cap_min"] for r in active)
    tot_max = sum(r["cap_max"] for r in active)
    by_src = {}
    for r in active:
        by_src[r["source"]] = by_src.get(r["source"], 0) + r["capacity"]
    print(f"容量（唯一地块，排除人工标记后）: 默认(垂直28) {tot_cap}；布局区间 {tot_min}–{tot_max}")
    print(f"  分来源: {by_src}；排除 {len(excluded)} 个地块（{sum(r['capacity'] for r in excluded)} 车位）")

    # 7) 输出 GeoJSON / CSV / 预览图 / meta（P2-15 相对路径）
    feats_utm = [{"type": "Feature", "geometry": mapping(lot["geom"]), "properties": r}
                 for lot, r in zip(lots, rows)]
    (OUT / "aoi_capacity_utm.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": UTM}},
                    "features": feats_utm}, ensure_ascii=False), encoding="utf-8")
    feats_wgs = [{"type": "Feature", "geometry": transform_geom(UTM, WGS, mapping(lot["geom"])), "properties": r}
                 for lot, r in zip(lots, rows)]
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
    for lot, r in zip(lots, rows):
        color = (0, 230, 110) if lot["source"] == "seg" else ((255, 150, 0) if lot["source"] == "osm" else (0, 180, 255))
        xy = [(float(x), float(y)) for x, y in lot["geom"].exterior.coords]
        px = [(int((x - wt.c) / wt.a), int((y - wt.f) / wt.e)) for x, y in xy]
        dr.line(px, fill=color, width=2)
        dr.text((px[0][0] + 3, px[0][1] + 3), f"{r['capacity']}", fill=color, font=font)
    img.save(OUT / "aoi_capacity_preview.png")

    # 8) 真值采集工单（唯一地块，含经纬度）
    truth_rows = []
    for lot, r in zip(lots, rows):
        lon, lat = wtransform(UTM, WGS, [r["cx"]], [r["cy"]])
        truth_rows.append({
            "lot_id": r["id"], "lon": round(lon[0], 6), "lat": round(lat[0], 6),
            "area_m2": r["area_m2"], "is_parking": "", "true_capacity": "",
            "layout_true": "", "notes": "", "collected_by": "", "date": "",
            "model_capacity": r["capacity"],
        })
    truth_rows.sort(key=lambda x: -float(x["area_m2"]))
    with open(OUT / "truth_collection_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["lot_id", "lon", "lat", "area_m2", "is_parking",
                                          "true_capacity", "layout_true", "notes",
                                          "collected_by", "date", "model_capacity"])
        w.writeheader()
        w.writerows(truth_rows)

    # 9) meta
    (OUT / "aoi_capacity_meta.json").write_text(json.dumps({
        "tif": TIF.name, "seg_weights": str(SEG_W.relative_to(ROOT)),
        "aoi": [col, row, size], "conf": CONF, "min_area_m2": MIN_AREA,
        "coef_table": COEF, "default_layout": DEFAULT_LAYOUT,
        "match_rule": {"min_overlap_ratio": MATCH_MIN_RATIO, "min_iou": MATCH_MIN_IOU},
        "counts": {"total": len(rows), "both": matched,
                   "seg_only": sum(1 for r in rows if r["source"] == "seg"),
                   "osm_only": sum(1 for r in rows if r["source"] == "osm")},
        "capacity_total_unique": tot_cap,
        "capacity_excluded": sum(r["capacity"] for r in excluded),
        "excluded_ids": [r["id"] for r in excluded],
        "capacity_interval_unique": [tot_min, tot_max],
        "capacity_by_source": by_src,
        "caveat": "布局为默认垂直式；cap_min/max 覆盖平行/斜列系数，属工程区间非统计置信区间；AOI 与标注训练区重叠，分割表现为 in-sample；真值未校准",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出 -> {OUT}")
    print("  aoi_capacity_preview.png / .csv / _utm.geojson / _wgs84.geojson / _meta.json / truth_collection_sheet.csv")


if __name__ == "__main__":
    main()