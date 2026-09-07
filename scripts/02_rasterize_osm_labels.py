"""OSM 停车场弱标签栅格化：Overpass JSON -> 二值掩膜 GeoTIFF / PNG。

读取 data/osm/chengdu_parking_raw.json（scripts/03_fetch_osm_parking.py 产出），
过滤掉 multi-storey / underground / rooftop（保留露天 surface 与未标注类型），
栅格化到与某张 GeoTIFF 影像一致的空间参考上，或按指定 bbox+分辨率生成独立标签。

用法:
  # 对齐到已有影像
  python scripts/02_rasterize_osm_labels.py data/osm/chengdu_parking_raw.json imagery.tif labels.tif
  # 指定经纬范围独立生成（west south east north, 分辨率米/像素）
  python scripts/02_rasterize_osm_labels.py data/osm/chengdu_parking_raw.json labels.tif --bbox 104.05 30.60 104.12 30.68 --res 0.5
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_parking_polygons(path):
    """返回露天停车场的 WGS84 多边形列表 [(exterior, [holes...]), ...]。"""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    data = json.load(open(path, encoding="utf-8"))
    polys = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if tags.get("parking") in ("multi-storey", "underground", "rooftop", "sheds"):
            continue  # 只保留露天
        if el["type"] == "way" and "geometry" in el:
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(pts) >= 4:
                polys.append(shape({"type": "Polygon", "coordinates": [pts]}))
        elif el["type"] == "relation" and "members" in el:
            rings = {}
            for m in el["members"]:
                if m["type"] == "way" and "geometry" in m:
                    rings.setdefault(m.get("role", "outer"), []).append(
                        [(p["lon"], p["lat"]) for p in m["geometry"]])
            for ring in rings.get("outer", []):
                if len(ring) >= 4:
                    polys.append(shape({"type": "Polygon", "coordinates": [ring]}))
    if not polys:
        raise SystemExit("未找到有效多边形")
    return unary_union(polys) if len(polys) > 1 else polys[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("osm_json", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--align-to", type=Path, default=None,
                    help="对齐到某张 GeoTIFF 的范围/坐标系")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--res", type=float, default=0.5, help="米/像素")
    ap.add_argument("--png", action="store_true", help="同时输出 PNG（给无坐标切片用）")
    args = ap.parse_args()

    import rasterio
    import rasterio.features
    import rasterio.warp
    from rasterio.transform import from_bounds
    from pyproj import Transformer

    geom = load_parking_polygons(args.osm_json)

    if args.align_to:
        with rasterio.open(args.align_to) as src:
            transform, width, height = src.transform, src.width, src.height
            crs = src.crs
    else:
        if not args.bbox:
            raise SystemExit("需要 --align-to 或 --bbox")
        w, s, e, n = args.bbox
        crs = "EPSG:4326"
        # bbox 为经纬度；0.5m/像素需换算成度
        lat_mid = (s + n) / 2
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_mid))
        px_w = args.res / m_per_deg_lon
        px_h = args.res / m_per_deg_lat
        width = int((e - w) / px_w)
        height = int((n - s) / px_h)
        transform = from_bounds(w, s, e, n, width, height)
        print(f"栅格尺寸: {width} x {height}")

    # OSM 是 WGS84，若目标 crs 不是 4326 则重投影
    from shapely.ops import transform as shp_transform
    if crs != "EPSG:4326":
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        geom = shp_transform(tr, geom)

    mask = rasterio.features.rasterize(
        [(geom, 1)], out_shape=(height, width),
        transform=transform, fill=0, dtype="uint8")

    with rasterio.open(args.output, "w", driver="GTiff", height=height,
                       width=width, count=1, dtype="uint8", crs=crs,
                       transform=transform) as dst:
        dst.write(mask, 1)
    cov = mask.mean() * 100
    print(f"完成: {args.output}  停车场覆盖率 {cov:.2f}%")
    if args.png:
        import cv2
        cv2.imwrite(str(args.output).replace(".tif", ".png"),
                    (mask * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
