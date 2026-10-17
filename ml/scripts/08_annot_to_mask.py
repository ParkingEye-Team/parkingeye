"""X-AnyLabeling / LabelMe JSON -> 影像坐标系掩膜（分割训练标签）。

输入：data/annot_aux/lot_*.png + 同名 .json（X-AnyLabeling 导出，shapes[].points
      为像素坐标，label='parking'），以及每张图对应的 .txt 元数据
      （含 col/row 偏移，即该 PNG 左上角在整幅影像中的像素位置）。
输出：整幅影像坐标下的分割标签 GeoTIFF + 抽样 PNG 供检查。

用法:
  python scripts/08_labelme_to_mask.py --annot-dir data/annot_aux \
      --tif <影像.tif> --out data/seg_manual/deadmask 等
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import Polygon as ShapelyPolygon, box as ShapelyBox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot-dir", type=Path, required=True)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--out-mask", type=Path, required=True,
                    help="输出掩膜 GeoTIFF 路径（整幅影像范围）")
    args = ap.parse_args()

    # 读影像元数据
    with rasterio.open(args.tif) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    # 收集标注：像素多边形 -> 影像全局像素多边形
    polygons = []  # [(label, [ [x,y]... 全局像素 ])]
    jsons = sorted(glob.glob(str(args.annot_dir / "*.json")))
    if not jsons:
        raise SystemExit(f"未找到标注 json: {args.annot_dir}/*.json")
    used_png = set()
    for jf in jsons:
        jf = Path(jf)
        png_path = args.annot_dir / (jf.stem + ".png")
        if not png_path.exists():
            print(f"[跳过] {jf.name}: 无对应 PNG")
            continue
        # 读 txt 元数据得 col/row 偏移（兼容 CRLF/同行多键值）
        txt_path = jf.with_suffix(".txt")
        col0 = row0 = 0
        if txt_path.exists():
            content = txt_path.read_text(encoding="utf-8")
            import re
            mcol = re.search(r"col=(\d+)", content)
            mrow = re.search(r"row=(\d+)", content)
            if mcol:
                col0 = int(mcol.group(1))
            if mrow:
                row0 = int(mrow.group(1))
        img = cv2.imread(str(png_path))
        if img is None:
            print(f"[跳过] {jf.name}: PNG 读取失败")
            continue
        ih, iw = img.shape[:2]
        used_png.add(png_path.name)
        data = json.loads(jf.read_text(encoding="utf-8"))
        n_poly = 0
        n_clipped = 0
        for shp in data.get("shapes", []):
            label = shp.get("label", "")
            pts = shp.get("points", [])
            if label != "parking" or len(pts) < 3:
                continue
            # 像素坐标 -> 全局像素坐标（+偏移）；越界部分裁剪而不是整条丢弃（P2-5）
            global_pts = [(x + col0, y + row0) for x, y in pts]
            try:
                poly = ShapelyPolygon(global_pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                clipped = poly.intersection(ShapelyBox(0, 0, width, height))
            except Exception:
                continue
            if clipped.is_empty:
                n_clipped += 1
                continue
            parts = list(clipped.geoms) if clipped.geom_type == "MultiPolygon" else [clipped]
            for part in parts:
                coords = [(x, y) for x, y in part.exterior.coords[:-1]]
                if len(coords) < 3:
                    continue
                polygons.append((label, coords))
                n_poly += 1
            if not poly.equals(clipped):
                n_clipped += 1
        print(f"{jf.name}: {n_poly} 个停车场多边形, 裁剪越界 {n_clipped} 个 (col0={col0}, row0={row0})")

    print(f"总计 {len(polygons)} 个停车场多边形, 来自 {len(used_png)} 张图")

    # 用 cv2.fillPoly 把像素坐标多边形直接画进掩膜（最可靠，无坐标系陷阱）
    mask = np.zeros((height, width), dtype="uint8")
    for label, pts in polygons:
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], 1)
    if not polygons:
        print("[警告] 无有效多边形，掩膜全零")

    # 写 GeoTIFF
    args.out_mask.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=height, width=width, count=1,
                   dtype="uint8", crs=crs, transform=transform, compress="lzw")
    with rasterio.open(args.out_mask, "w", **profile) as dst:
        dst.write(mask, 1)
    print(f"掩膜写入: {args.out_mask}  停车场占比 {mask.mean() * 100:.3f}%")


if __name__ == "__main__":
    main()
