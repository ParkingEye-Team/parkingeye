"""吉林一号预标注生成：JL1 切片 + DOTA 模型预测 -> X-AnyLabeling 可导入的预标注 JSON。

流程（模型辅助标注，加速 3-5 倍）：
  1. 从 JL1 大图按 OSM 停车场位置裁 1024 窗口（也可指定纯道路窗口）
  2. DOTA 基线模型预测（conf 0.25，已验证"框出来的都是车"）
  3. 导出：切片 JPG + 同名 .json（X-AnyLabeling rotation 格式，带 score）
  4. 标注组在 X-AnyLabeling 打开：删误检 + 补漏检 -> 导出 -> 微调

输出：data/jl1_annot/images|labels（供标注）、data/jl1_annot/preannot_raw（原始预标注备份）

用法:
  ./.venv/Scripts/python.exe ml/scripts/20_preannot_jl1.py --tif <影像.tif> --n 12 [--conf 0.25]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import transform as wtransform
from PIL import Image as PILImage
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DET_W = ROOT / "runs/obb/dota_vehicle_obb/weights/best.pt"
OSM = ROOT / "data/osm/chengdu_parking_raw.json"
OUT = ROOT / "data/jl1_annot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--n", type=int, default=12, help="生成窗口数")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tag", default=None, help="输出子目录标签(默认用影像名)")
    args = ap.parse_args()

    tif = Path(args.tif)
    tag = args.tag or tif.parent.name[:24]
    img_dir = OUT / tag / "images"
    json_dir = OUT / tag / "preannot_raw"
    img_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    det = YOLO(str(DET_W))
    els = [e for e in json.load(open(OSM, encoding="utf-8"))["elements"] if "geometry" in e]
    lons = [np.mean([g["lon"] for g in e["geometry"]]) for e in els]
    lats = [np.mean([g["lat"] for g in e["geometry"]]) for e in els]

    made = 0
    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        xs, ys = wtransform("EPSG:4326", src.crs, lons, lats)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if made >= args.n:
                break
            col, row = ~src.transform * (x, y)
            col, row = int(col), int(row)
            if not (0 <= col < W and 0 <= row < H):
                continue
            c0 = max(0, min(col - 512, W - 1024))
            r0 = max(0, min(row - 512, H - 1024))
            win = rasterio.windows.Window(c0, r0, 1024, 1024)
            arr = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0)).astype(np.uint8)
            if (arr.max(axis=2) < 8).mean() > 0.3:  # 死区过多跳过
                continue
            r = det(arr, conf=args.conf, iou=0.55, imgsz=1024, max_det=2000, verbose=False)[0]
            pts = [] if r.obb is None else r.obb.xyxyxyxy.cpu().numpy()
            confs = [] if r.obb is None else r.obb.conf.cpu().numpy()
            stem = f"jl1_{tag}_osm{i:03d}"
            # 保存切片（原图, 不增强——训练用原图；标注时可在软件中自行放大）
            PILImage.fromarray(arr).save(img_dir / f"{stem}.jpg", "JPEG", quality=95)
            # 导出 X-AnyLabeling JSON（rotation, 带 score 便于标注组按置信度排查）
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
                "imageHeight": 1024, "imageWidth": 1024, "description": "",
            }
            (json_dir / f"{stem}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            made += 1
            print(f"{stem}: 预标注 {len(shapes)} 框 (col={c0}, row={r0})")
    print(f"\n完成 {made} 个窗口 -> {OUT / tag}")
    print("标注组操作: 见 data/jl1_annot/README.md")


if __name__ == "__main__":
    main()
