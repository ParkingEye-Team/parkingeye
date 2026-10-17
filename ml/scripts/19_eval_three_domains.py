"""三域评估：GE 留出域 / 冬季吉林一号 / 夏季吉林一号。

对给定权重：
  1. GE 留出域: model.val() 在 data/ge_yolo_tiled/val（17 切片, 真留出）-> mAP
  2. 冬季窗口: annot_aux 3 个 lot 窗口, conf0.25 imgsz1024 max_det2000 -> 检出数
  3. 夏季窗口: 夏季景内 OSM 停车场窗口(前 15 个) -> 检出数
结果写入 data/infer/three_domain_eval/<权重名>.json + 打印摘要

用法:
  ./.venv/Scripts/python.exe ml/scripts/19_eval_three_domains.py <权重路径> [标签]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import transform as wtransform
from ultralytics import YOLO
from osm_utils import is_surface_parking

ROOT = Path(__file__).resolve().parents[2]
WINTER_TIF = ROOT / "data/imagery/JL1KF01C/JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH" \
             / "JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH.tif"
SUMMER_TIF = ROOT / "data/imagery/JL1KF01C_PMSL1_20240825114907_200295580_102_0009_001_L3D_PSH" \
             / "JL1KF01C_PMSL1_20240825114907_200295580_102_0009_001_L3D_PSH.tif"
OSM = ROOT / "data/osm/chengdu_parking_raw.json"
GE_YAML = ROOT / "data/ge_yolo_tiled/ge_obb_tiled.yaml"
OUT = ROOT / "data/infer/three_domain_eval"
IMGSZ, MAX_DET, CONF = 1024, 2000, 0.25


def ge_val(model):
    """GE 留出域 mAP（17 切片）。"""
    try:
        m = model.val(data=str(GE_YAML), split="val", imgsz=IMGSZ, batch=2, workers=0, verbose=False)
        return {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                "P": round(float(m.box.mp), 4), "R": round(float(m.box.mr), 4)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def window_detections(model, tif, windows):
    ns = []
    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        for c0, r0 in windows:
            c0 = max(0, min(c0, W - IMGSZ)); r0 = max(0, min(r0, H - IMGSZ))
            arr = np.transpose(src.read([1, 2, 3], window=rasterio.windows.Window(c0, r0, IMGSZ, IMGSZ)), (1, 2, 0)).astype(np.uint8)
            if (arr.max(axis=2) < 8).mean() > 0.5:
                continue
            # 注意（doc/11 P0-3）：arr 为 rasterio RGB；DOTA 旧权重训练时 R/B 被交换，
            # 推理必须继续传 RGB 对齐，不要改成 BGR（除非重训 DOTA）。
            r = model(arr, conf=CONF, iou=0.55, imgsz=IMGSZ, max_det=MAX_DET, verbose=False)[0]
            ns.append(0 if r.obb is None else len(r.obb))
    return ns


def main():
    weights = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else Path(weights).parent.parent.name
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    res = {"weights": weights, "tag": tag}

    # 1) GE 留出
    res["ge_val"] = ge_val(model)
    print(f"[GE 留出] {res['ge_val']}")

    # 2) 冬季窗口（annot_aux 前3个）
    w_windows = []
    import glob
    for tp in sorted(glob.glob(str(ROOT / "data/annot_aux/*.txt")))[:3]:
        c = open(tp, encoding="utf-8", errors="ignore").read()
        mcol = re.search(r"col=(\d+)", c)
        mrow = re.search(r"row=(\d+)", c)
        if not (mcol and mrow):
            print(f"[跳过] {tp}: 缺少 col/row 偏移")
            continue
        w_windows.append((int(mcol.group(1)), int(mrow.group(1))))
    w_ns = window_detections(model, WINTER_TIF, w_windows)
    res["winter"] = {"windows": len(w_ns), "total": int(sum(w_ns)), "counts": [int(x) for x in w_ns]}
    print(f"[冬季窗口] {res['winter']}")

    # 3) 夏季窗口（OSM 前15个落在夏季景内的）
    els = [e for e in json.load(open(OSM, encoding="utf-8"))["elements"]
           if "geometry" in e and is_surface_parking(e.get("tags", {}))]
    lons = [np.mean([g["lon"] for g in e["geometry"]]) for e in els]
    lats = [np.mean([g["lat"] for g in e["geometry"]]) for e in els]
    with rasterio.open(SUMMER_TIF) as src:
        H, W = src.height, src.width
        xs, ys = wtransform("EPSG:4326", src.crs, lons, lats)
        s_windows = []
        for x, y in zip(xs, ys):
            col, row = ~src.transform * (x, y)
            col, row = int(col), int(row)
            if 0 <= col < W and 0 <= row < H:
                s_windows.append((col - 512, row - 512))
            if len(s_windows) >= 15:
                break
    s_ns = window_detections(model, SUMMER_TIF, s_windows)
    res["summer"] = {"windows": len(s_ns), "total": int(sum(s_ns)), "median": float(np.median(s_ns)) if s_ns else 0,
                     "counts": [int(x) for x in s_ns]}
    print(f"[夏季窗口] {res['summer']}")

    out = OUT / f"{tag}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
