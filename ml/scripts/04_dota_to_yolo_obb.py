"""DOTA 原始包 -> YOLO-OBB 训练集转换（只取车辆类，合并为单一 vehicle 类）。

流程: v1.0 影像 zip + v1.5 标注（8坐标旋转框）
  1. 解压该 split 的标注 zip 到临时目录
  2. 逐张读影像（支持多 part zip，跨 zip 去重——DOTA 影像名全局唯一）
  3. 1024x1024 滑动切片（重叠 200），对每个切片：
     - 实例多边形裁剪到切片范围（shapely 求交）
     - 取裁剪后多边形的 minAreaRect 作为旋转框
     - 过滤过小实例、越界框
     - 同一张影像的多切片名带坐标后缀
  4. 输出 Ultralytics OBB 目录结构 + data yaml

用法:
  python scripts/04_dota_to_yolo_obb.py --train-dir C:/baidunetdiskdownload/train \
      --val-dir C:/baidunetdiskdownload/val --out data/dota_yolo --tile 1024 --overlap 200
"""
import argparse
import glob
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

KEEP = {"small-vehicle", "large-vehicle", "car"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_annotations(txt_path: Path):
    """返回 [(poly[8], cls, diff), ...]，忽略头部注释行。"""
    objs = []
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10 or parts[0].startswith("imagesource") \
                    or parts[0].startswith("gsd"):
                continue
            try:
                poly = [float(v) for v in parts[:8]]
                cls = parts[8]
                diff = int(parts[9])
            except (ValueError, IndexError):
                continue
            objs.append((poly, cls, diff))
    return objs


def extract_labels(src_dir: Path, label_version: str) -> Path:
    """解压 split 的标注到临时目录，返回含 P*.txt 的目录。自动发现标注 zip。

    不同镜像目录文件名不统一（labelTxt.zip / DOTA-v1.5_train.zip / *_hbb.zip 等），
    规则：取 *.zip 中非 *_hbb* 且非 *Task2_gt* 的那个（hbb 是水平框，Task2_gt 是旧任务）。
    """
    cands = sorted(src_dir.glob(f"labelTxt-{label_version}/*.zip"))
    cands = [c for c in cands if "_hbb" not in c.stem.lower()
             and "task2_gt" not in c.stem.lower()
             and "task1" not in c.stem.lower()]
    if not cands:
        raise SystemExit(f"未找到标注 zip: {src_dir / 'labelTxt-' + label_version}")
    label_zip = cands[0]
    tmp = Path(tempfile.mkdtemp(prefix="dota_labels_"))
    with zipfile.ZipFile(label_zip) as z:
        for n in z.namelist():
            if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                z.extract(n, tmp)
    return tmp


def iter_zip_images(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            ext = Path(name).suffix.lower()
            if ext not in IMG_EXT or name.startswith("__MACOSX"):
                continue
            arr = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                yield Path(name).stem, img


def polygon_area(poly):
    """shoelace 面积（像素）。"""
    xs, ys = poly[0::2], poly[1::2]
    return 0.5 * abs(sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i]
                         for i in range(4)))


def clip_and_rect(poly, x, y, tile_w, tile_h):
    """平移多边形到切片坐标，与切片框求交，返回 minAreaRect 4 角点（像素坐标）或 None。

    poly: [x1,y1,x2,y2,x3,y3,x4,y4] 全局像素坐标
    返回 4x2 float32 数组（0..tile 内）。
    """
    import shapely.geometry as sg
    # 平移
    p = sg.Polygon([(poly[i] - x, poly[i + 1] - y) for i in range(0, 8, 2)])
    tile_box = sg.box(0, 0, tile_w, tile_h)
    inter = p.intersection(tile_box)
    if inter.is_empty:
        return None
    if inter.geom_type == "MultiPolygon":
        inter = max(inter.geoms, key=lambda g: g.area)
    if inter.geom_type != "Polygon" or inter.area < 16:
        return None
    # 取外环点转 minAreaRect
    ex = np.array(inter.exterior.coords, dtype=np.float32)
    rect = cv2.minAreaRect(ex.reshape(-1, 1, 2))
    return cv2.boxPoints(rect)


def process_split(split, src_dir, label_version, out, tile, overlap, seen):
    """处理 train 或 val。seen 用于跨 split/zip 去重。返回 (切片数, 跳过数)。"""
    label_dir = extract_labels(src_dir, label_version)
    img_zips = sorted(glob.glob(str(src_dir / "images" / "*.zip")))
    n_tiles = n_skip = 0
    for zip_path in img_zips:
        print(f"  [{split}] {Path(zip_path).name}")
        for stem, img in iter_zip_images(Path(zip_path)):
            if stem in seen:
                continue
            seen.add(stem)
            ann_path = label_dir / f"{stem}.txt"
            if not ann_path.exists():
                n_skip += 1
                continue
            objs = read_annotations(ann_path)
            h, w = img.shape[:2]
            step = tile - overlap
            for y in range(0, max(h - overlap, 1), step):
                for x in range(0, max(w - overlap, 1), step):
                    y2, x2 = min(y + tile, h), min(x + tile, w)
                    if y2 - y < tile // 2 or x2 - x < tile // 2:
                        continue
                    tile_img = img[y:y2, x:x2]
                    th, tw = tile_img.shape[:2]
                    lines = []
                    for poly, cls, diff in objs:
                        if cls not in KEEP:
                            continue
                        if polygon_area(poly) < 40:  # 全局太小，切片后必过小
                            continue
                        box = clip_and_rect(poly, x, y, tw, th)
                        if box is None:
                            continue
                        if cv2.contourArea(box) < 20:
                            continue
                        # 归一化并校验范围
                        norm = box / [tw, th]
                        if norm.min() < -0.02 or norm.max() > 1.02:
                            continue
                        line = "0 " + " ".join(f"{c:.4f}" for p in norm for c in p)
                        if line not in lines:
                            lines.append(line)
                    if not lines:
                        continue
                    fname = f"{stem}__{x}_{y}"
                    cv2.imwrite(str(out / "images" / split / f"{fname}.png"),
                                cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB))
                    (out / "labels" / split / f"{fname}.txt").write_text(
                        "\n".join(lines), encoding="utf-8")
                    n_tiles += 1
    shutil.rmtree(label_dir, ignore_errors=True)
    return n_tiles, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/dota_yolo"))
    ap.add_argument("--label-version", choices=["v1.0", "v1.5"], default="v1.5")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    seen = set()
    for split, d in (("train", args.train_dir), ("val", args.val_dir)):
        nt, ns = process_split(split, Path(d), args.label_version, out,
                               args.tile, args.overlap, seen)
        print(f"[{split}] 切片 {nt} 张, 无标注跳过 {ns} 张")

    yaml_path = out / "dota_obb.yaml"
    yaml_path.write_text(
        f"path: {str(out).replace(chr(92), '/')}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n  0: vehicle\n", encoding="utf-8")
    print(f"完成。YAML: {yaml_path}")
    print(f"   训练切片: {len(list((out/'images'/'train').glob('*.png')))}")
    print(f"   验证切片: {len(list((out/'images'/'val').glob('*.png')))}")


if __name__ == "__main__":
    main()
