"""吉林一号预标注生成：JL1 切片 + DOTA 模型预测 -> X-AnyLabeling 可导入的预标注 JSON。

流程（模型辅助标注，加速 3-5 倍）：
  1. 从 JL1 大图按 OSM 停车场位置裁 1024 窗口（做窗口去重，避免同一片车场重复出窗）
  2. DOTA 基线模型预测（conf 0.25，已验证"框出来的都是车"；通道保持 RGB，见 P0-3 说明）
  3. 导出：切片 JPG + 同名 .json（X-AnyLabeling rotation 格式，带 score）
  4. 标注组在 X-AnyLabeling 打开：删误检 + 补漏检 -> 导出 -> 微调
  5. 输出 manifest.csv，记录每个窗口的 TIF/col/row/OSM ID/框数/均值分，便于追溯

输出：data/jl1_annot/<tag>/{images,preannot_raw,manifest.csv}

用法:
  ./.venv/Scripts/python.exe ml/scripts/20_preannot_jl1.py --tif <影像.tif> --n 12 [--conf 0.25] [--tag xxx]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import transform as wtransform
from PIL import Image as PILImage
from ultralytics import YOLO
from osm_utils import is_surface_parking

ROOT = Path(__file__).resolve().parents[2]
DET_W = ROOT / "runs/obb/dota_vehicle_obb/weights/best.pt"
OSM = ROOT / "data/osm/chengdu_parking_raw.json"
OUT = ROOT / "data/jl1_annot"

TILE = 1024
MAX_WINDOW_IOU = 0.30   # 同场景窗口 IoU 超过该值则跳过（避免重复标注/训练泄漏）


def window_iou(a, b):
    """两个同尺寸正方形窗口的 IoU；窗口为 (col, row, size)。"""
    ax, ay, s = a
    bx, by, _ = b
    ix = max(0, min(ax + s, bx + s) - max(ax, bx))
    iy = max(0, min(ay + s, by + s) - max(ay, by))
    inter = ix * iy
    union = 2 * s * s - inter
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--n", type=int, default=12, help="最大接受窗口数")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tag", default=None, help="输出子目录标签(默认用影像文件名)")
    args = ap.parse_args()

    tif = Path(args.tif)
    tag = args.tag or tif.stem[:40]
    img_dir = OUT / tag / "images"
    json_dir = OUT / tag / "preannot_raw"
    img_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / tag / "manifest.csv"

    det = YOLO(str(DET_W))
    els = [e for e in json.load(open(OSM, encoding="utf-8"))["elements"]
           if "geometry" in e and is_surface_parking(e.get("tags", {}))]
    lons = [np.mean([g["lon"] for g in e["geometry"]]) for e in els]
    lats = [np.mean([g["lat"] for g in e["geometry"]]) for e in els]

    accepted = []   # (col, row, size)
    rows = []
    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        size = min(TILE, H, W)
        xs, ys = wtransform("EPSG:4326", src.crs, lons, lats)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if len(accepted) >= args.n:
                break
            col, row = ~src.transform * (x, y)
            col, row = int(col), int(row)
            if not (0 <= col < W and 0 <= row < H):
                continue
            c0 = max(0, min(col - size // 2, W - size))
            r0 = max(0, min(row - size // 2, H - size))
            cand = (c0, r0, size)
            # 窗口去重（P0-4）：与已接受窗口重合过多则跳过
            if any(window_iou(cand, a) > MAX_WINDOW_IOU for a in accepted):
                continue
            arr = np.transpose(src.read([1, 2, 3], window=rasterio.windows.Window(c0, r0, size, size)), (1, 2, 0)).astype(np.uint8)
            if (arr.max(axis=2) < 8).mean() > 0.3:  # 死区过多跳过
                continue
            # DOTA 旧权重保持 RGB 输入（见 doc/11 P0-3）；不要改成 BGR
            r = det(arr, conf=args.conf, iou=0.55, imgsz=size, max_det=2000, verbose=False)[0]
            pts = [] if r.obb is None else r.obb.xyxyxyxy.cpu().numpy()
            confs = [] if r.obb is None else r.obb.conf.cpu().numpy()
            stem = f"jl1_{tag}_osm{i:03d}"
            PILImage.fromarray(arr).save(img_dir / f"{stem}.jpg", "JPEG", quality=95)
            shapes = []
            for p, cf in zip(pts, confs):
                shapes.append({
                    "label": "vehicle",
                    "points": [[float(px), float(py)] for px, py in p],
                    "group_id": None, "shape_type": "rotation", "flags": {},
                    "description": "", "difficult": False, "direction": 0,
                    "score": round(float(cf), 3), "attributes": {}, "kie_linking": [],
                })
            doc = {
                "version": "3.0.0", "flags": {}, "checked": False, "shapes": shapes,
                "imagePath": f"{stem}.jpg", "imageData": None,
                "imageHeight": int(size), "imageWidth": int(size), "description": "",
            }
            (json_dir / f"{stem}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            accepted.append(cand)
            mean_score = float(np.mean(confs)) if len(confs) else 0.0
            rows.append({
                "window": stem, "tif": tif.name, "col": c0, "row": r0, "size": size,
                "osm_index": i, "osm_id": els[i].get("id", ""),
                "n_boxes": len(shapes), "mean_score": round(mean_score, 3),
            })
            print(f"{stem}: 预标注 {len(shapes)} 框 (col={c0}, row={r0})")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["window", "tif", "col", "row", "size",
                                          "osm_index", "osm_id", "n_boxes", "mean_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成 {len(rows)} 个唯一窗口 -> {OUT / tag}")
    print(f"manifest: {manifest_path}")
    print("标注组操作: 见 data/jl1_annot/README.md")


if __name__ == "__main__":
    main()