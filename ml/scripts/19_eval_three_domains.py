"""三域评估：GE 留出域 / 冬季吉林一号 / 夏季吉林一号。

对给定权重：
  1. GE 留出域: model.val() 在 data/ge_yolo_tiled/val（17 切片, 真留出）-> mAP
  2. 冬季窗口: annot_aux 3 个 lot 窗口, conf0.25 imgsz1024 max_det2000 -> 检出数
  3. 夏季窗口: 夏季景内 OSM 停车场窗口(前 15 个) -> 检出数

通道约定（doc/records/audit-2026-10-17.md P0-3 + doc/records/audit-2026-10-18.md P1-3）：
  - DOTA 旧权重训练时 R/B 交换，JL1 窗口推理必须传 RGB；
  - GE v2/v3 权重训练用正常 BGR，JL1 窗口推理应传 BGR；
  - 因此本脚本按权重路径自动判断，也可用 --channel 显式指定。

用法:
  ./.venv/Scripts/python.exe ml/scripts/19_eval_three_domains.py <权重路径> [标签]
  ./.venv/Scripts/python.exe ml/scripts/19_eval_three_domains.py <权重> ge_v3 --channel bgr
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import transform as wtransform
from ultralytics import YOLO
from osm_utils import is_surface_parking

from safe_io import write_json_safe

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


def window_detections(model, tif, windows, channel="rgb"):
    ns = []
    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        for c0, r0 in windows:
            c0 = max(0, min(c0, W - IMGSZ)); r0 = max(0, min(r0, H - IMGSZ))
            arr = np.transpose(src.read([1, 2, 3], window=rasterio.windows.Window(c0, r0, IMGSZ, IMGSZ)), (1, 2, 0)).astype(np.uint8)
            if (arr.max(axis=2) < 8).mean() > 0.5:
                continue
            # rasterio 给的是 RGB；GE v2/v3 训练用 BGR，需要翻转
            inp = np.ascontiguousarray(arr[:, :, ::-1]) if channel == "bgr" else arr
            r = model(inp, conf=CONF, iou=0.55, imgsz=IMGSZ, max_det=MAX_DET, verbose=False)[0]
            ns.append(0 if r.obb is None else len(r.obb))
    return ns


def infer_channel(weights):
    """按权重路径/同目录 args.yaml 推断通道；无法判断时默认 RGB 并告警。"""
    s = str(weights).lower()
    if "ge_finetune" in s or "ge_yolo" in s:
        return "bgr"
    if "dota" in s:
        return "rgb"
    args_yaml = Path(weights).parent.parent / "args.yaml"
    if args_yaml.exists():
        text = args_yaml.read_text(encoding="utf-8", errors="ignore").lower()
        if "ge_finetune" in text or "ge_yolo" in text:
            return "bgr"
        if "dota" in text:
            return "rgb"
    print("[警告] 无法从权重路径/args.yaml 判断通道，默认 RGB；请用 --channel 指定")
    return "rgb"


def main():
    ap = argparse.ArgumentParser(description="三域评估")
    ap.add_argument("weights")
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--channel", choices=["auto", "rgb", "bgr"], default="auto")
    args = ap.parse_args()
    weights = args.weights
    tag = args.tag or Path(weights).parent.parent.name
    channel = infer_channel(weights) if args.channel == "auto" else args.channel
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    res = {"weights": weights, "tag": tag, "channel": channel}

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
    w_ns = window_detections(model, WINTER_TIF, w_windows, channel)
    res["winter"] = {"windows": len(w_ns), "total": int(sum(w_ns)), "counts": [int(x) for x in w_ns]}
    print(f"[冬季窗口/{channel}] {res['winter']}")

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
    s_ns = window_detections(model, SUMMER_TIF, s_windows, channel)
    res["summer"] = {"windows": len(s_ns), "total": int(sum(s_ns)), "median": float(np.median(s_ns)) if s_ns else 0,
                     "counts": [int(x) for x in s_ns]}
    print(f"[夏季窗口/{channel}] {res['summer']}")

    out = OUT / f"{tag}_{channel}.json"
    write_json_safe(out, res, tag="three_domain")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
