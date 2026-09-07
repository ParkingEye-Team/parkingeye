"""停车场分割训练数据准备：影像+掩膜配对切片（正负样本平衡）。

输入：
  - GeoTIFF 影像（如 JL1KF01C 冬季 0.5m）
  - OSM 停车场多边形 GeoJSON/Overpass JSON（正样本掩膜来源）
做法：
  1. 将 OSM 停车场多边形栅格化为与影像同 geotransform 的整幅掩膜（uint8）
  2. 正样本切片：取"包含停车场像素的网格单元"切 512x512
  3. 负样本切片：在无停车场的网格单元中随机/均匀采样同数量块（负样本）
  4. 输出 (image, mask) 对到 data/seg_{tag}/images|masks/{split}

用法:
  python scripts/06_prep_seg_data.py --tif <影像.tif> --osm <停车场.json> \
      --out data/seg_kf01c --tile 512 --train-neg-ratio 1.0
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, box
from shapely.ops import unary_union


def load_parking_geoms(osm_path, src, tr=None):
    """读 Overpass JSON，过滤地下/立体，重投影到影像 CRS，返回合并多边形。"""
    data = json.load(open(osm_path, encoding="utf-8"))
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    polys = []
    for el in data["elements"]:
        if el["type"] != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        if tags.get("parking") in ("multi-storey", "underground", "rooftop", "sheds"):
            continue
        pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(pts) < 4:
            continue
        utm = [tr.transform(lon, lat) for lon, lat in pts]
        p = shape({"type": "Polygon", "coordinates": [[(x, y) for x, y in utm]]})
        # 与影像范围求交
        p = p.intersection(box(*src.bounds))
        if not p.is_empty and p.geom_type in ("Polygon", "MultiPolygon"):
            polys.append(p)
    return unary_union(polys) if len(polys) > 1 else (polys[0] if polys else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--osm", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/seg_kf01c"))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="负样本:正样本 比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "masks" / sp).mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.tif) as src:
        # 1) 整幅掩膜栅格化
        union = load_parking_geoms(args.osm, src)
        if union is None:
            raise SystemExit("无有效停车场多边形")
        mask = rasterize([(union, 1)], out_shape=(src.height, src.width),
                         transform=src.transform, fill=0, dtype="uint8")
        print(f"掩膜覆盖率: {mask.mean()*100:.3f}%  (影像 {src.width}x{src.height})")

        # 2) 网格扫描，收集正/负单元
        tile = args.tile
        pos_units, neg_units = [], []
        for y in range(0, src.height - tile, tile):
            for x in range(0, src.width - tile, tile):
                m = mask[y:y + tile, x:x + tile]
                frac = m.mean()
                if frac > 0.002:          # 单元内含少量停车场像素即正样本
                    pos_units.append((x, y, frac))
                elif frac == 0 and len(neg_units) < int(len(pos_units)) * 10:
                    # 负样本池（先只收集，实际挑选在平衡时进行）
                    neg_units.append((x, y))

        # 正样本上限：内存与训练成本考虑（每张 512x512x3 uint8 ≈ 0.8MB）
        # 但正样本本来就不多，直接全用；负样本按比例采样
        random.shuffle(neg_units)
        n_pos, n_neg = len(pos_units), int(len(pos_units) * args.neg_ratio)
        selected_neg = neg_units[:n_neg]
        print(f"正样本单元: {n_pos}, 负样本选取: {len(selected_neg)}/{len(neg_units)}")

        # 3) 写切片
        def write_xy(x, y, split):
            win = rasterio.windows.Window(x, y, tile, tile)
            img = src.read([1, 2, 3], window=win)
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
            m = mask[y:y + tile, x:x + tile]
            name = f"{x}_{y}"
            cv2.imwrite(str(out / "images" / split / f"{name}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "masks" / split / f"{name}.png"), m * 255)

        # 划分 train/val：按单元划分（同影像区域同 split 防止泄漏）
        all_units = [(x, y, 1) for x, y, _ in pos_units] + \
                    [(x, y, 0) for x, y in selected_neg]
        random.shuffle(all_units)
        n_val = int(len(all_units) * args.val_split)
        for i, (x, y, _) in enumerate(all_units):
            split = "val" if i < n_val else "train"
            write_xy(x, y, split)

        n_tot = len(all_units)
        print(f"切片总数: {n_tot} (train {n_tot-n_val}, val {n_val}) -> {out}")


if __name__ == "__main__":
    main()
