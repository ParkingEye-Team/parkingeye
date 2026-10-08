"""GE 大图 + YOLO-OBB 标签 -> 1024 切片（供微调，避免整图缩得太小车辆退化）。

输入：data/ge_yolo/train/images|labels（YOLO-OBB 归一化标签，大图）
输出：data/ge_yolo_tiled/train/images|labels（1024×1024 切片，带 200 重叠）
      切片内坐标重新归一化；完全落在切片内的框才保留（避免切碎的目标）。

用法:
  ./.venv/Scripts/python.exe ml/scripts/15_ge_tile_obb.py
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/ge_yolo" / "train"
OUT = ROOT / "data/ge_yolo_tiled" / "train"
TILE = 1024
OVERLAP = 200
STRIDE = TILE - OVERLAP
MIN_BOX_AREA_RATIO = 0.15  # 框面积占切片面积比例下限(相对框本身, 实际过滤切碎目标用交并比)


def main():
    imgs = sorted((SRC / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"无输入图: {SRC / 'images'}")
    for sub in ("images", "labels"):
        d = OUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n_tiles = n_boxes = 0
    for ip in imgs:
        # PIL 读图（cv2 读不了中文路径）；转 BGR 供 cv2.imwrite
        img = cv2.cvtColor(np.asarray(Image.open(ip).convert("RGB")), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        labs = []
        lp = SRC / "labels" / (ip.stem + ".txt")
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 9:
                    continue
                xy = np.array(parts[1:], dtype=np.float64).reshape(4, 2)
                xy[:, 0] *= W
                xy[:, 1] *= H
                labs.append((int(parts[0]), xy))
        # 滑窗
        for y0 in range(0, H - TILE + 1, STRIDE):
            for x0 in range(0, W - TILE + 1, STRIDE):
                tile = img[y0:y0 + TILE, x0:x0 + TILE]
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
                if keep:
                    stem = f"{ip.stem}_x{x0}_y{y0}"
                    Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)).save(
                        str(OUT / "images" / f"{stem}.jpg"), "JPEG", quality=95)
                    (OUT / "labels" / f"{stem}.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
                    n_tiles += 1
                    n_boxes += len(keep)
    print(f"完成：{len(imgs)} 张大图 -> {n_tiles} 切片 / {n_boxes} 框  -> {OUT}")


if __name__ == "__main__":
    main()
