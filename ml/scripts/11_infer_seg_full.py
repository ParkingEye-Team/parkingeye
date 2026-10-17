"""停车场分割全图推理（滑动窗口）脚本。

对整幅 GeoTIFF 影像做滑窗推理，输出：预测掩膜（停车场=1）GeoTIFF + 可视化叠加 PNG。

注意：分割模型训练时使用 BGR（cv2 约定），推理前必须把 rasterio 的 RGB 转为 BGR，
否则命中率会大幅下降（2026-10-17 审查：同一批夏季窗口 RGB 26% vs BGR 65%）。

用法:
  python ml/scripts/11_infer_seg_full.py --model <best.pt> --tif <影像> \
      --out data/infer/full_mask.tif --tile 512 --overlap 64 --device cpu
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from ultralytics import YOLO


def tile_positions(size: int, tile: int, step: int):
    """生成滑窗起点，并补上右/下边缘的最后一个窗口（避免漏掉边缘条带）。"""
    if size <= tile:
        return [0]
    pos = list(range(0, size - tile, step))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/infer/full_mask.tif"))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = YOLO(str(args.model))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.tif) as src:
        height, width = src.height, src.width
        profile = src.profile
        # 预测累积与计数（重叠区取平均）
        acc = np.zeros((height, width), dtype=np.float32)
        cnt = np.zeros((height, width), dtype=np.uint8)
        step = args.tile - args.overlap
        n_win = 0
        ys = tile_positions(height, args.tile, step)
        xs = tile_positions(width, args.tile, step)
        for y in ys:
            for x in xs:
                win = rasterio.windows.Window(x, y, args.tile, args.tile)
                img = src.read([1, 2, 3], window=win)
                rgb = np.transpose(img, (1, 2, 0)).astype(np.uint8)
                # 跳过死区
                if rgb.mean() < 5:
                    continue
                # rasterio 读为 RGB；Ultralytics numpy 输入按 BGR 解释 -> 必须转换
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                r = model.predict(bgr, conf=args.conf, imgsz=args.tile,
                                  device=args.device, verbose=False)[0]
                mask = np.zeros((args.tile, args.tile), dtype=np.float32)
                if r.masks is not None:
                    m = r.masks.data.cpu().numpy()
                    for mi in m:
                        mi = cv2.resize(mi, (args.tile, args.tile))
                        mask = np.maximum(mask, mi)
                acc[y:y + args.tile, x:x + args.tile] += mask
                cnt[y:y + args.tile, x:x + args.tile] += 1
                n_win += 1
                if n_win % 50 == 0:
                    print(f"已推理 {n_win} 窗, 进度 y={y}/{height} x={x}/{width}",
                          flush=True)
        # 平均
        cnt[cnt == 0] = 1
        pred = (acc / cnt > 0.4).astype(np.uint8)
        print(f"预测停车场占比: {pred.mean() * 100:.3f}%   ({n_win} 窗)")

        profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(pred, 1)
        print(f"掩膜写入: {out}")


if __name__ == "__main__":
    main()
