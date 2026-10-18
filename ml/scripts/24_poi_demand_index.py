"""POI 需求代理指标：把高德 POI 汇总到容量地块，生成空间评估表。

方法：
  - 读取 data/poi/amap_poi_wgs84.geojson（23 号脚本产物）与容量地块；
  - 跳过 excluded=True 或 exclude_lots.csv 中的地块（非停车场不计入需求排名）；
  - 在 AOI 局部米制平面上，统计每个地块 buffer（默认 500m）内各类 POI 数量；
  - 加权得到 demand_weighted，并在有效地块间做百分位排名 demand_index（0-100）；
  - pressure = demand_weighted / capacity，表示"每车位对应的需求代理点数"（相对指标）。

重要口径：
  - POI 密度只是**需求代理**，不等于真实停车需求/供需缺口；
  - 权重是工程假设（住宅/办公/商业为主），不是从数据学出来的；
  - 未考虑竞争停车供给、道路连通性、收费与管制，结论只能用于相对比较；
  - 地块缓冲相互重叠，逐地块计数不可相加。

输出：
  - data/poi/poi_demand_by_lot.csv       逐地块 POI 计数、加权需求、排名、压力
  - data/poi/aoi_capacity_poi_wgs84.geojson  容量地块 + 需求指标（供 Web 看板）
  - data/poi/poi_demand_meta.json        参数、权重、口径说明
  （覆盖已有文件前自动备份到 data/capacity/backup/）

用法:
  ./.venv/Scripts/python.exe ml/scripts/24_poi_demand_index.py
  ./.venv/Scripts/python.exe ml/scripts/24_poi_demand_index.py --buffer 800
"""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from shapely.geometry import shape, Point
from shapely.ops import transform as shp_transform

from safe_io import run_id, write_csv_safe, write_json_safe

ROOT = Path(__file__).resolve().parents[2]
CAP_GJ = ROOT / "data/capacity/aoi_capacity_wgs84.geojson"
POI_GJ = ROOT / "data/poi/amap_poi_wgs84.geojson"
OUT_DIR = ROOT / "data/poi"
CAP_DIR = ROOT / "data/capacity"

CATEGORIES = ["residential", "office", "retail", "dining", "leisure",
              "public_service", "hotel"]
# 工程假设权重：住宅（夜间）、办公（白天通勤）、商业餐饮（访客）为主。
WEIGHTS = {
    "residential": 0.30,
    "office": 0.25,
    "retail": 0.15,
    "dining": 0.15,
    "leisure": 0.08,
    "public_service": 0.04,
    "hotel": 0.03,
}


def local_projector(lng0, lat0):
    m_lat = 111320.0
    m_lng = 111320.0 * math.cos(math.radians(lat0))
    return lambda lng, lat: ((lng - lng0) * m_lng, (lat - lat0) * m_lat)


def rank_percentile(values):
    """返回 0-100 百分位（并列取平均名次），最小值为 0，最大值为 100。"""
    n = len(values)
    if n <= 1:
        return [50.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank / (n - 1) * 100.0
        i = j + 1
    return ranks


def load_excluded():
    ids = set()
    p = CAP_DIR / "exclude_lots.csv"
    if p.exists():
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("lot_id"):
                    ids.add(r["lot_id"].strip())
    return ids


def main():
    ap = argparse.ArgumentParser(description="POI 需求代理指标")
    ap.add_argument("--buffer", type=float, default=500, help="地块缓冲区半径（米），默认 500")
    ap.add_argument("--poi", default=str(POI_GJ), help="POI GeoJSON 路径")
    args = ap.parse_args()

    cap = json.loads(CAP_GJ.read_text(encoding="utf-8"))
    poi = json.loads(Path(args.poi).read_text(encoding="utf-8"))
    features = cap["features"]
    exclude_ids = load_excluded()

    # AOI 中心作为局部投影原点（兼容 Polygon/MultiPolygon）
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for f in features:
        g = shape(f["geometry"])
        a, b, c, d = g.bounds
        minx, miny, maxx, maxy = min(minx, a), min(miny, b), max(maxx, c), max(maxy, d)
    proj = local_projector((minx + maxx) / 2, (miny + maxy) / 2)

    poi_pts = []
    for f in poi["features"]:
        lng, lat = f["geometry"]["coordinates"]
        x, y = proj(lng, lat)
        poi_pts.append((Point(x, y), f["properties"].get("category", "other")))

    rows = []
    skipped = []
    for f in features:
        props = f["properties"]
        if props.get("excluded") or props.get("id") in exclude_ids:
            skipped.append(props.get("id"))
            continue
        lot_poly = shp_transform(lambda x, y, z=None: proj(x, y), shape(f["geometry"]))
        zone = lot_poly.buffer(args.buffer)

        counts = {c: 0 for c in CATEGORIES}
        for pt, cat in poi_pts:
            if cat in counts and zone.contains(pt):
                counts[cat] += 1
        total = sum(counts.values())
        weighted = sum(counts[c] * WEIGHTS[c] for c in CATEGORIES)
        rows.append({
            "lot_id": props["id"],
            "area_m2": props.get("area_m2"),
            "capacity": props.get("capacity"),
            "cap_min": props.get("cap_min"),
            "cap_max": props.get("cap_max"),
            "poi_total": total,
            **{f"poi_{c}": counts[c] for c in CATEGORIES},
            "demand_weighted": round(weighted, 3),
            "demand_per_space": round(weighted / props["capacity"], 4)
            if props.get("capacity") else None,
        })

    rid = run_id()
    if not rows:
        print("警告：没有有效地块，写出空产物")
        write_csv_safe(OUT_DIR / "poi_demand_by_lot.csv", [], tag="poi")
        write_json_safe(OUT_DIR / "poi_demand_meta.json",
                        {"run_id": rid, "lots": 0, "excluded": skipped}, tag="poi")
        return

    idx = rank_percentile([r["demand_weighted"] for r in rows])
    for r, v in zip(rows, idx):
        r["demand_index"] = round(v, 1)

    rows.sort(key=lambda r: r["demand_index"], reverse=True)
    csv_path = OUT_DIR / "poi_demand_by_lot.csv"
    write_csv_safe(csv_path, rows, tag="poi")

    # 把指标写回容量 GeoJSON（排除地块保留原样，不写需求字段，Web 端过滤）
    by_id = {r["lot_id"]: r for r in rows}
    for f in features:
        r = by_id.get(f["properties"]["id"])
        if r:
            f["properties"].update({
                "poi_total": r["poi_total"],
                "demand_weighted": r["demand_weighted"],
                "demand_index": r["demand_index"],
                "demand_per_space": r["demand_per_space"],
            })
    write_json_safe(OUT_DIR / "aoi_capacity_poi_wgs84.geojson", cap,
                    indent=None, tag="poi")

    meta = {
        "run_id": rid,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "buffer_m": args.buffer,
        "weights": WEIGHTS,
        "lots": len(rows),
        "excluded_lots": skipped,
        "category_counts_sum_over_lots": {c: sum(r[f"poi_{c}"] for r in rows) for c in CATEGORIES},
        "top5_demand": [{"lot_id": r["lot_id"], "demand_index": r["demand_index"],
                         "poi_total": r["poi_total"], "capacity": r["capacity"]}
                        for r in rows[:5]],
        "caveat": "POI 密度仅为需求代理指标，不等于真实供需；权重为工程假设；"
                  "未考虑竞争供给、路网与管制；demand_index 是有效地块内的相对百分位；"
                  "地块缓冲相互重叠，逐地块计数不可相加。",
    }
    write_json_safe(OUT_DIR / "poi_demand_meta.json", meta, tag="poi")

    print(f"地块 {len(rows)} | 排除 {len(skipped)} | 缓冲 {args.buffer}m | POI {len(poi_pts)}")
    print(f"输出: {csv_path}")
    print("\n需求指数 Top 5:")
    for r in rows[:5]:
        print(f"  {r['lot_id']}  index={r['demand_index']:5.1f}  POI={r['poi_total']:4d}  "
              f"容量={r['capacity']}  每车位需求={r['demand_per_space']}")


if __name__ == "__main__":
    main()
