"""基于人工精标掩膜的语义分割训练数据准备（正负样本平衡）。

输入：整幅掩膜 GeoTIFF（manual_mask.tif，人工标注）+ 影像 GeoTIFF + 死区掩膜。
做法：
  1. 死区掩膜排除纯黑区域（不产生样本）
  2. 正样本：掩膜>0 的 512x512 网格单元
  3. 负样本：掩膜==0 且非死区的网格单元（按比例采样）
输出：data/seg_kf01c_manual/{images,masks}/{train,val}

用法:
  python ml/scripts/09_prep_seg_manual.py --tif <影像> --mask <掩膜> \
      --deadmask <死区> --out data/seg_kf01c_manual --tile 512 --neg-ratio 1.0
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--deadmask", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/seg_kf01c_manual"))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "masks" / sp).mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.mask) as msrc:
        mask = msrc.read(1)
        height, width = mask.shape
    with rasterio.open(args.tif) as isrc:
        pass

    if args.deadmask and Path(args.deadmask).exists():
        with rasterio.open(args.deadmask) as dsrc:
            dead = dsrc.read(1).astype(bool)
    else:
        dead = None

    tile = args.tile
    pos, neg = [], []
    # 扫描网格（步长 tile，无重叠，保证无重复）
    for y in range(0, height - tile, tile):
        for x in range(0, width - tile, tile):
            if dead is not None and dead[y:y + tile, x:x + tile].mean() > 0.5:
                continue  # 死区
            m = mask[y:y + tile, x:x + tile]
            frac = (m > 0).mean()
            if frac > 0.01:
                pos.append((x, y))
            elif frac == 0:
                neg.append((x, y))

    random.shuffle(neg)
    n_neg = min(len(neg), int(len(pos) * args.neg_ratio) if pos else 300)
    neg = neg[:n_neg]
    print(f"正样本: {len(pos)}, 负样本: {len(neg)}")

    if not pos:
        raise SystemExit("无正样本块（掩膜可能太小或太稀疏）")

    # 划分 train/val（正负都按比例）
    all_units = [(x, y) for x, y in pos] + neg
    random.shuffle(all_units)
    n_val = int(len(all_units) * args.val_split)

    with rasterio.open(args.tif) as isrc:
        for i, (x, y) in enumerate(all_units):
            win = rasterio.windows.Window(x, y, tile, tile)
            img = isrc.read([1, 2, 3], window=win)
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
            m = mask[y:y + tile, x:x + tile]
            split = "val" if i < n_val else "train"
            name = f"{x}_{y}"
            cv2.imwrite(str(out / "images" / split / f"{name}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "masks" / split / f"{name}.png"), m * 255)

    n_tot = len(all_units)
    print(f"切片总数: {n_tot} (train {n_tot-n_val}, val {n_val}) -> {out}")


if __name__ == "__main__":
    main()
