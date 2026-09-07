"""生成停车场人工标注辅助图（供标注组在 QGIS/图上勾勒边界）。

对影像内每个"可用"(非死区)的 OSM 停车场，抠 1024x1024 增强图（CLAHE），
文件名带坐标与面积，方便标注组对照。同时输出一个总览 GeoJSON（含死区标记），
标注组可在 QGIS 中叠底影像直接勾勒修正。

用法:
  python scripts/07_gen_annot_aux.py --tif <影像> --osm <停车场json> \
      --deadmask <死区掩膜> --out data/annot_aux
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--osm", type=Path, required=True)
    ap.add_argument("--deadmask", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/annot_aux"))
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--margin", type=int, default=200)  # 地块外扩像素
    args = ap.parse_args()

    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dead_arr = None
    dead_profile = None

    with rasterio.open(args.tif) as src:
        if args.deadmask:
            with rasterio.open(args.deadmask) as dm:
                dead_arr = dm.read(1)

        data = json.load(open(args.osm, encoding="utf-8"))
        lots = []
        for el in data["elements"]:
            if el["type"] != "way" or "geometry" not in el:
                continue
            tags = el.get("tags", {})
            if tags.get("parking") in ("multi-storey", "underground", "rooftop"):
                continue
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            # 转投影坐标后用影像 bounds（同为投影）判断是否在内
            utm = [tr.transform(a, b) for a, b in pts]
            b = src.bounds
            if not all(b.left < x < b.right and b.bottom < y < b.top
                       for x, y in utm):
                continue
            lon = sum(x for x, y in pts) / len(pts)
            lat = sum(y for x, y in pts) / len(pts)
            # 经纬度 -> 投影
            utm = [tr.transform(a, b) for a, b in pts]
            xs, ys = [p[0] for p in utm], [p[1] for p in utm]
            minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
            area_m2 = (maxx - minx) * (maxy - miny)  # 近似外接盒面积
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            row, col = src.index(cx, cy)
            # 判断是否死区
            if dead_arr is not None:
                dead = dead_arr[max(0, row - 40):row + 40,
                                max(0, col - 40):col + 40].mean() > 0.5
            else:
                dead = False
            lots.append({
                "lon": lon, "lat": lat, "cx": cx, "cy": cy,
                "row": row, "col": col, "area_m2": area_m2,
                "dead": dead, "bbox_utm": (minx, miny, maxx, maxy),
            })

        usable = [l for l in lots if not l["dead"]]
        dead_l = [l for l in lots if l["dead"]]
        print(f"总停车场 {len(lots)}: 可用 {len(usable)}, 死区 {len(dead_l)}")
        print(f"按面积排序取前 {min(50, len(usable))} 个生成标注图")

        # 按面积从大到小，取前 50 个可用
        usable.sort(key=lambda l: -l["area_m2"])
        selected = usable[:50]
        m = args.margin
        tile = args.tile
        for i, lot in enumerate(selected):
            r0 = max(0, lot["row"] - tile // 2)
            c0 = max(0, lot["col"] - tile // 2)
            win = rasterio.windows.Window(c0, r0, tile, tile)
            arr = src.read([1, 2, 3], window=win)
            img = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
            # CLAHE 增强
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            img2 = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                                cv2.COLOR_LAB2RGB)
            fname = f"lot_{i + 1:02d}_c{c0}_r{r0}_a{lot['area_m2'] / 1e4:.1f}ha"
            cv2.imwrite(str(out / f"{fname}.png"),
                        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
            # 记录元数据
            (out / f"{fname}.txt").write_text(
                f"center_lon={lot['lon']}\ncenter_lat={lot['lat']}\n"
                f"utm_x={lot['cx']:.1f}\nutm_y={lot['cy']:.1f}\n"
                f"area_m2={lot['area_m2']:.0f}\n"
                f"col={c0} row={r0} tile={tile}\n", encoding="utf-8")

        print(f"生成 {len(selected)} 张标注辅助图 -> {out}")


if __name__ == "__main__":
    main()
