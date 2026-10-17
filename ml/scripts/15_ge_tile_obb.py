"""GE 大图 + YOLO-OBB 标签 -> 1024 切片（供微调，避免整图缩得太小车辆退化）。

输入：data/ge_yolo/{train,val}/images|labels（YOLO-OBB 归一化标签，大图）
输出：data/ge_yolo_tiled/{train,val}/images|labels（1024×1024 切片，带 200 重叠）
      切片内坐标重新归一化；完全落在切片内的框才保留（避免切碎的目标）。
      边缘补窗（P0-5）覆盖整图；每张图保留少量纯背景切片（P1-1）。

用法:
  ./.venv/Scripts/python.exe ml/scripts/15_ge_tile_obb.py
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "data/ge_yolo"
OUT_ROOT = ROOT / "data/ge_yolo_tiled"
TILE = 1024
OVERLAP = 200
STRIDE = TILE - OVERLAP
MAX_EMPTY_PER_IMAGE = 2   # 每张图最多保留的纯背景切片数


def tile_positions(size: int, tile: int, stride: int):
    """滑窗起点，补右/下边缘最后一个窗口（P0-5）。"""
    if size <= tile:
        return [0]
    pos = list(range(0, size - tile, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def tile_split(split: str):
    src = SRC_ROOT / split
    out = OUT_ROOT / split
    imgs = sorted((src / "images").glob("*.jpg"))
    if not imgs:
        print(f"[{split}] 无输入图，跳过")
        return 0, 0
    for sub in ("images", "labels"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n_tiles = n_boxes = n_empty = 0
    for ip in imgs:
        # PIL 读图（cv2 读不了中文路径）；转 BGR 供 cv2.imwrite
        img = cv2.cvtColor(np.asarray(Image.open(ip).convert("RGB")), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        labs = []
        lp = src / "labels" / (ip.stem + ".txt")
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 9:
                    continue
                xy = np.array(parts[1:], dtype=np.float64).reshape(4, 2)
                xy[:, 0] *= W
                xy[:, 1] *= H
                labs.append((int(parts[0]), xy))
        # 滑窗（含边缘补窗）
        empty_kept = 0
        for y0 in tile_positions(H, TILE, STRIDE):
            for x0 in tile_positions(W, TILE, STRIDE):
                tile = img[y0:y0 + TILE, x0:x0 + TILE]
                if tile.shape[0] != TILE or tile.shape[1] != TILE:
                    continue
                keep = []
                for cls, xy in labs:
                    # 框完全在切片内
                    if not (np.all(xy[:, 0] >= x0) and np.all(xy[:, 0] <= x0 + TILE)
                            and np.all(xy[:, 1] >= y0) and np.all(xy[:, 1] <= y0 + TILE)):
                        continue
                    # 目标不能太小(原始尺寸过滤, 防噪点)
                    w = np.linalg.norm(xy[1] - xy[0]); h = np.linalg.norm(xy[2] - xy[1])
                    if min(w, h) < 8 or w * h < 200:
                        continue
                    rel = (xy - np.array([x0, y0])) / TILE
                    keep.append(f"{cls} " + " ".join(f"{v:.6f}" for v in rel.flatten()))
                stem = f"{ip.stem}_x{x0}_y{y0}"
                if keep:
                    Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)).save(
                        str(out / "images" / f"{stem}.jpg"), "JPEG", quality=95)
                    (out / "labels" / f"{stem}.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
                    n_tiles += 1
                    n_boxes += len(keep)
                elif empty_kept < MAX_EMPTY_PER_IMAGE:
                    # 保留纯背景切片（P1-1），空标签文件即背景
                    if (tile.max() < 8):  # 全黑死区不保留
                        continue
                    Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)).save(
                        str(out / "images" / f"{stem}.jpg"), "JPEG", quality=95)
                    (out / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
                    empty_kept += 1
                    n_tiles += 1
                    n_empty += 1
    print(f"[{split}] {len(imgs)} 张大图 -> {n_tiles} 切片 / {n_boxes} 框 (其中背景 {n_empty})")
    return n_tiles, n_boxes


def main():
    t1, b1 = tile_split("train")
    t2, b2 = tile_split("val")
    print(f"完成：train {t1} 切片 / {b1} 框；val {t2} 切片 / {b2} 框 -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
