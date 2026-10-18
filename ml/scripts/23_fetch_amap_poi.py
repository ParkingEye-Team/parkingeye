"""高德 POI 拉取：为容量地块构建需求代理指标的原始 POI 数据。

数据流：
  data/capacity/aoi_capacity_wgs84.geojson（24 个地块，WGS84）
    -> AOI 外扩 buffer -> 规则网格（默认 1200m）
    -> 每个网格中心转 GCJ-02（高德坐标系）后调用 place/around
    -> 按 7 类兴趣点分页拉取 -> 按 POI id 去重
    -> 坐标转回 WGS84（与影像/OSM/容量产物一致）
    -> data/poi/amap_poi_raw.csv + amap_poi_wgs84.geojson + meta.json

坐标约定（重要）：
  - 高德 API 的 location 参数与返回坐标均为 GCJ-02；
  - 项目的容量/OSM 产物是 WGS84；
  - 本脚本保存 lng_gcj/lat_gcj 与 lng_wgs/lat_wgs 两套坐标，后续分析只用 WGS84。

输出：
  - data/poi/amap_poi_raw.csv         逐 POI（含分类、网格、双坐标）
  - data/poi/amap_poi_wgs84.geojson   POI 点图层（Web 看板/空间分析用）
  - data/poi/amap_poi_meta.json       请求数、分类计数、截断与配额信息
  - data/poi/cache/*.json             单网格单分类缓存（--resume 时复用）

用法:
  ./.venv/Scripts/python.exe ml/scripts/23_fetch_amap_poi.py
  ./.venv/Scripts/python.exe ml/scripts/23_fetch_amap_poi.py --buffer 1000 --grid 1000 --radius 800
  ./.venv/Scripts/python.exe ml/scripts/23_fetch_amap_poi.py --categories retail,dining
  ./.venv/Scripts/python.exe ml/scripts/23_fetch_amap_poi.py --no-resume   # 强制重拉

注意：
  - AMAP_KEY 从项目根目录 .env 读取，不打印完整值；
  - 高德 POI 数据仅限本项目内部评估使用，勿公开分发/入库（.gitignore 已忽略 data/poi/）；
  - POI 密度只是需求代理，不等于真实停车供需。
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from shapely.geometry import shape

from safe_io import write_csv_safe, write_json_safe, write_text_safe

ROOT = Path(__file__).resolve().parents[2]
CAP_GJ = ROOT / "data/capacity/aoi_capacity_wgs84.geojson"
OUT_DIR = ROOT / "data/poi"
CACHE_DIR = OUT_DIR / "cache"
API = "https://restapi.amap.com/v3/place/around"

# 高德 typecode 分类（| 表示同一需求类别下的多个类别）
CATEGORIES = {
    "residential": "120300",          # 住宅区
    "office": "120200|170000",        # 楼宇 / 公司企业
    "retail": "060000",               # 购物服务
    "dining": "050000",               # 餐饮服务
    "leisure": "080000",              # 体育休闲
    "public_service": "090000|140000",  # 医疗保健 / 科教文化
    "hotel": "100000",                # 住宿服务
}

QUOTA_INFOCODES = {
    "10001": "Key 无效或已过期",
    "10003": "日访问量超出配额",
    "10004": "访问过于频繁（QPS 超限）",
    "10009": "Key 与平台类型不匹配（应为 Web 服务）",
    "10010": "IP 白名单限制",
    "10044": "账号日调用量超限",
    "10045": "IP 日调用量超限",
    "10048": "账号 QPS 超限",
}

# ---------------------------------------------------------------------------
# GCJ-02 <-> WGS84（高德坐标系转换，标准公开算法）
# ---------------------------------------------------------------------------
_A = 6378245.0
_EE = 0.00669342162296594323


def _out_of_china(lng, lat):
    return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)


def _transform_lat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
        0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
        0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lng, lat):
    if _out_of_china(lng, lat):
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (_A / sqrtmagic * math.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat


def gcj02_to_wgs84(lng, lat):
    if _out_of_china(lng, lat):
        return lng, lat
    wlng, wlat = lng, lat
    for _ in range(3):
        glng, glat = wgs84_to_gcj02(wlng, wlat)
        wlng += lng - glng
        wlat += lat - glat
    return wlng, wlat


# ---------------------------------------------------------------------------
# 环境与输入
# ---------------------------------------------------------------------------
def load_key():
    env_path = ROOT / ".env"
    if not env_path.exists():
        sys.exit("未找到 .env，请先配置 AMAP_KEY")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("AMAP_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key:
                return key
    sys.exit(".env 中没有 AMAP_KEY")


def mask_key(key):
    """只保留末 4 位；不再暴露首 4 位（doc/records/audit-2026-10-18.md P1-6）。"""
    return "***" + key[-4:] if len(key) >= 4 else "***"


def load_lots():
    gj = json.loads(CAP_GJ.read_text(encoding="utf-8"))
    lots = []
    for f in gj["features"]:
        # Polygon/MultiPolygon 统一用 shapely；当前数据全是 Polygon，但未来会扩
        geom = shape(f["geometry"])
        lots.append({
            "id": f["properties"]["id"],
            "capacity": f["properties"].get("capacity"),
            "geom": geom,
        })
    return lots


def aoi_bbox(lots):
    minx = min(lot["geom"].bounds[0] for lot in lots)
    miny = min(lot["geom"].bounds[1] for lot in lots)
    maxx = max(lot["geom"].bounds[2] for lot in lots)
    maxy = max(lot["geom"].bounds[3] for lot in lots)
    return minx, miny, maxx, maxy


def make_grid(bbox, buffer_m, grid_m):
    """在 WGS84 近似平面（局部米制）上生成覆盖 AOI 外扩 buffer 的网格中心。"""
    minx, miny, maxx, maxy = bbox
    midlat = (miny + maxy) / 2
    m_per_deg_lat = 111320.0
    m_per_deg_lng = 111320.0 * math.cos(math.radians(midlat))
    minx -= buffer_m / m_per_deg_lng
    maxx += buffer_m / m_per_deg_lng
    miny -= buffer_m / m_per_deg_lat
    maxy += buffer_m / m_per_deg_lat
    nx = max(1, math.ceil((maxx - minx) * m_per_deg_lng / grid_m))
    ny = max(1, math.ceil((maxy - miny) * m_per_deg_lat / grid_m))
    step_x = (maxx - minx) / nx
    step_y = (maxy - miny) / ny
    tiles = []
    for iy in range(ny):
        for ix in range(nx):
            tiles.append({
                "ix": ix, "iy": iy,
                "lng": minx + (ix + 0.5) * step_x,
                "lat": miny + (iy + 0.5) * step_y,
            })
    return tiles, {"nx": nx, "ny": ny, "bbox_padded_wgs84": [minx, miny, maxx, maxy]}


# ---------------------------------------------------------------------------
# 拉取
# ---------------------------------------------------------------------------
def fetch_tile_category(session, key, tile, tile_idx, cat, typecodes,
                        radius, max_pages, resume, cache_sig):
    # 缓存名包含参数签名，改 buffer/grid/radius/max_pages 后不会误用旧缓存
    cache_path = CACHE_DIR / f"t{tile_idx:03d}_{cat}_{cache_sig}.json"
    if resume and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("cache_sig") == cache_sig:
            return data["pois"], data["pages"], data["count"], False, True

    glng, glat = wgs84_to_gcj02(tile["lng"], tile["lat"])
    pois, pages = [], 0
    total_count = 0
    truncated = False
    for page in range(1, max_pages + 1):
        params = {
            "key": key,
            "location": f"{glng:.6f},{glat:.6f}",
            "radius": radius,
            "types": typecodes,
            "offset": 25,
            "page": page,
            "extensions": "base",
            "sortrule": "distance",
        }
        for attempt in range(3):
            try:
                resp = session.get(API, params=params, timeout=20)
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"请求失败 tile={tile_idx} cat={cat} page={page}: {exc}")
                time.sleep(1.5 * (attempt + 1))

        status = str(payload.get("status"))
        infocode = str(payload.get("infocode"))
        if status != "1":
            if infocode in QUOTA_INFOCODES:
                sys.exit(f"高德 API 拒绝请求：{QUOTA_INFOCODES[infocode]}（infocode={infocode}）")
            raise RuntimeError(f"高德错误 tile={tile_idx} cat={cat}: {payload.get('info')} ({infocode})")

        pages += 1
        total_count = int(payload.get("count", 0))
        batch = payload.get("pois") or []
        for p in batch:
            loc = (p.get("location") or "").split(",")
            if len(loc) != 2:
                continue
            try:
                plng, plat = float(loc[0]), float(loc[1])
            except ValueError:
                continue
            wlng, wlat = gcj02_to_wgs84(plng, plat)
            pois.append({
                "poi_id": p.get("id"),
                "name": p.get("name"),
                "type": p.get("type"),
                "typecode": p.get("typecode"),
                "address": p.get("address"),
                "lng_gcj": round(plng, 7),
                "lat_gcj": round(plat, 7),
                "lng_wgs": round(wlng, 7),
                "lat_wgs": round(wlat, 7),
                "category": cat,
                "tile": tile_idx,
            })
        if len(batch) < 25 or page * 25 >= total_count:
            break
        if page == max_pages and total_count > page * 25:
            truncated = True
        time.sleep(0.12)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "pois": pois, "pages": pages, "count": total_count,
        "truncated": truncated, "tile": tile_idx, "category": cat,
        "cache_sig": cache_sig,
    }, ensure_ascii=False), encoding="utf-8")
    return pois, pages, total_count, truncated, False


def main():
    ap = argparse.ArgumentParser(description="按 AOI 网格拉取高德 POI")
    ap.add_argument("--buffer", type=float, default=800, help="AOI 外扩距离（米），默认 800")
    ap.add_argument("--grid", type=float, default=1200, help="网格边长（米），默认 1200")
    ap.add_argument("--radius", type=float, default=None,
                    help="单次 around 查询半径（米），默认取网格对角线一半")
    ap.add_argument("--max-pages", type=int, default=40, help="单网格单分类最大翻页数（25 条/页）")
    ap.add_argument("--categories", default="all",
                    help="逗号分隔的分类名，默认 all；可选：" + ",".join(CATEGORIES))
    ap.add_argument("--no-resume", action="store_true", help="忽略缓存，强制重新拉取")
    args = ap.parse_args()

    cats = list(CATEGORIES) if args.categories == "all" else [c.strip() for c in args.categories.split(",")]
    unknown = [c for c in cats if c not in CATEGORIES]
    if unknown:
        sys.exit(f"未知分类: {unknown}；可选: {list(CATEGORIES)}")

    radius = args.radius or round(args.grid * math.sqrt(2) / 2)
    cache_sig = hashlib.md5(
        f"{args.buffer}|{args.grid}|{radius}|{args.max_pages}".encode()).hexdigest()[:8]
    key = load_key()
    lots = load_lots()
    bbox = aoi_bbox(lots)
    tiles, grid_meta = make_grid(bbox, args.buffer, args.grid)
    print(f"Key {mask_key(key)} | 地块 {len(lots)} | 网格 {len(tiles)} "
          f"({grid_meta['nx']}x{grid_meta['ny']}) | 半径 {radius}m | 分类 {len(cats)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "ParkingEye-SRTP/1.0"})

    seen = {}
    per_cat = {c: 0 for c in cats}
    requests_made = 0
    cache_hits = 0
    truncated_queries = []
    t0 = time.time()

    for tile_idx, tile in enumerate(tiles):
        for cat in cats:
            pois, pages, count, truncated, from_cache = fetch_tile_category(
                session, key, tile, tile_idx, cat, CATEGORIES[cat],
                radius, args.max_pages, not args.no_resume, cache_sig)
            if not from_cache:
                requests_made += pages
            else:
                cache_hits += 1
            if truncated:
                truncated_queries.append({"tile": tile_idx, "category": cat, "count": count})
            for p in pois:
                pid = p["poi_id"]
                if pid and pid not in seen:
                    seen[pid] = p
                    per_cat[cat] += 1
        print(f"  网格 {tile_idx + 1}/{len(tiles)} 完成，累计唯一 POI {len(seen)}，请求 {requests_made}")

    rows = list(seen.values())
    csv_path = OUT_DIR / "amap_poi_raw.csv"
    write_csv_safe(csv_path, rows, fieldnames=[
        "poi_id", "name", "type", "typecode", "address",
        "lng_gcj", "lat_gcj", "lng_wgs", "lat_wgs", "category", "tile"], tag="poi")

    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lng_wgs"], r["lat_wgs"]]},
            "properties": {k: r[k] for k in ("poi_id", "name", "type", "typecode",
                                             "address", "category")},
        } for r in rows],
    }
    write_json_safe(OUT_DIR / "amap_poi_wgs84.geojson", gj, indent=None, tag="poi")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "高德开放平台 place/around（Web 服务）",
        "key_masked": mask_key(key),
        "input": str(CAP_GJ.relative_to(ROOT)),
        "bbox_wgs84": bbox,
        "buffer_m": args.buffer,
        "grid_m": args.grid,
        "grid": grid_meta,
        "radius_m": radius,
        "max_pages": args.max_pages,
        "cache_sig": cache_sig,
        "categories": {c: {"typecodes": CATEGORIES[c], "unique_pois": per_cat[c]} for c in cats},
        "requests": requests_made,
        "cache_hits": cache_hits,
        "unique_pois": len(rows),
        "truncated_queries": truncated_queries,
        "elapsed_s": round(time.time() - t0, 1),
        "category_dedup_note": "同一 POI 若命中多个查询类别，只计入首次命中的类别；分类合计可能低于交叉类别真实数。",
        "caveat": "POI 密度仅为需求代理指标，不等于真实停车供需；高德坐标已转 WGS84，"
                  "转换残差约 0.5-2m（GCJ-02 逆变换近似）；数据仅限内部使用，不入库。",
    }
    write_json_safe(OUT_DIR / "amap_poi_meta.json", meta, tag="poi")

    print(f"\n完成：唯一 POI {len(rows)} | 网络请求 {requests_made} | 缓存命中 {cache_hits} | "
          f"用时 {meta['elapsed_s']}s")
    for c in cats:
        print(f"  {c}: {per_cat[c]}")
    if truncated_queries:
        print(f"  警告：{len(truncated_queries)} 个查询达到翻页上限，计数可能偏低")
    print(f"输出: {csv_path}")


if __name__ == "__main__":
    main()
