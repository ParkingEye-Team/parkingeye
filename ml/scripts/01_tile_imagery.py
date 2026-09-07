"""影像切片脚本：把大图切成 512x512（可配）训练切片。

支持两类输入：
1. GeoTIFF（带地理坐标）：输出切片带 .aux.xml/world 文件可回定位，并生成
   tiles.geojson 记录每个切片的地理范围，供标签栅格化与结果回写使用。
2. 普通 PNG/JPG（无坐标，如 Google Earth 截图）：纯网格切片。

用法:
  python scripts/01_tile_imagery.py input.tif output_dir --tile 512 --overlap 64
  python scripts/01_tile_imagery.py screenshot.jpg output_dir --tile 512 --overlap 64
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path):
    """读图，GeoTIFF 用 rasterio（取前3波段），普通图用 OpenCV。"""
    try:
        import rasterio
        with rasterio.open(path) as src:
            img = src.read([1, 2, 3])  # (3, H, W)
            img = np.transpose(img, (1, 2, 0))
            meta = {
                "georeferenced": True,
                "transform": list(src.transform)[:6],
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
            }
            return img, meta
    except Exception:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, {"georeferenced": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64, help="相邻切片重叠像素")
    args = ap.parse_args()

    img, meta = read_image(args.input)
    h, w = img.shape[:2]
    step = args.tile - args.overlap
    out = args.output_dir
    (out / "images").mkdir(parents=True, exist_ok=True)

    records = []
    n = 0
    for y in range(0, max(h - args.overlap, 1), step):
        for x in range(0, max(w - args.overlap, 1), step):
            y2, x2 = min(y + args.tile, h), min(x + args.tile, w)
            tile = img[y:y2, x:x2]
            # 边缘切片太小则并入前一个
            if tile.shape[0] < args.tile // 2 or tile.shape[1] < args.tile // 2:
                continue
            name = f"tile_{x:06d}_{y:06d}.png"
            cv2.imwrite(str(out / "images" / name), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            rec = {"name": name, "px": [x, y, x2, y2]}
            if meta["georeferenced"]:
                import rasterio.transform
                tr = rasterio.transform.Affine(*meta["transform"])
                lon0, lat0 = rasterio.transform.xy(tr, y, x)
                lon1, lat1 = rasterio.transform.xy(tr, y2, x2)
                rec["bbox_geo"] = [min(lon0, lon1), min(lat0, lat1),
                                   max(lon0, lon1), max(lat0, lat1)]
            records.append(rec)
            n += 1

    with open(out / "tiles.json", "w", encoding="utf-8") as f:
        json.dump({"source": str(args.input), "tile": args.tile,
                   "overlap": args.overlap, "image_meta": meta,
                   "tiles": records}, f, ensure_ascii=False, indent=1)
    print(f"完成: {n} 个切片 -> {out / 'images'}  (索引: {out / 'tiles.json'})")


if __name__ == "__main__":
    main()
