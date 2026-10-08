"""成都冬季影像车辆检测复测（修正版 2，替代初版）。

修正点（对应 doc/05 第七节）：
  1. OBB 结果从 res.obb 读取（初版误用 res.boxes，导致旋转框恒记 0）；
  2. 覆盖 conf 0.25 / 0.10 / 0.05 三档阈值；
  3. 每档保存预测叠加图（原图 + 旋转框）到 data/infer/winter_recheck_v2/，供人工复核；
  4. 结果写入 CSV，含权重、图、档位、检出数、conf 范围。

用法:
  ./.venv/Scripts/python.exe ml/scripts/13_recheck_chengdu_winter.py [最多lot数]
"""
import csv
import glob
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.windows
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
TIF = ROOT / "data/imagery/JL1KF01C/JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH" \
      / "JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH.tif"
WEIGHTS = ROOT / "runs/obb/dota_vehicle_obb/weights/best.pt"
ANNOT_DIR = ROOT / "data/annot_aux"
OUT_DIR = ROOT / "data/infer" / "winter_recheck_v2"
CONFS = [0.25, 0.10, 0.05]

MAX_LOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # 默认复测前 3 个 lot
# 推理分辨率用窗口原生尺寸（1024）。默认 640 会把 0.5m 影像的车进一步缩小、显著降召回
# （实测同一窗口 640→64 检出 / 1024→125 检出）。
IMGSZ = 1024
MAX_DET = 2000  # 默认 300 会在密集车场截断检出，故放宽


def main():
    txts = sorted(glob.glob(str(ANNOT_DIR / "*.txt")))[:MAX_LOTS]
    if not txts:
        raise SystemExit("annot_aux 下没有 *.txt")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    fieldnames = ["lot", "variant", "conf", "vehicles", "conf_max", "conf_min"]
    csv_path = OUT_DIR / "winter_recheck_v2.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        with rasterio.open(TIF) as src:
            for tp in txts:
                content = open(tp, encoding="utf-8", errors="ignore").read()
                mcol = re.search(r"col=(\d+)", content)
                mrow = re.search(r"row=(\d+)", content)
                mtile = re.search(r"tile=(\d+)", content)
                if not (mcol and mrow):
                    print(f"跳过（缺偏移）: {Path(tp).name}")
                    continue
                c0, r0 = int(mcol.group(1)), int(mrow.group(1))
                tile = int(mtile.group(1)) if mtile else 1024
                name = Path(tp).stem
                for variant, arr in _read_variants(src, c0, r0, tile):
                    for conf in CONFS:
                        res = model(arr, conf=conf, imgsz=IMGSZ, max_det=MAX_DET, verbose=False)[0]
                        obb = res.obb
                        n = 0 if obb is None else len(obb)
                        confs = None if obb is None else obb.conf
                        row = {
                            "lot": name, "variant": variant, "conf": conf,
                            "vehicles": n,
                            "conf_max": round(float(confs.max()), 3) if n else "",
                            "conf_min": round(float(confs.min()), 3) if n else "",
                        }
                        w.writerow(row)
                        # 叠加图：有检出才画框；无检出也存原图便于对照
                        if n:
                            vis = _draw(arr, obb)
                        else:
                            vis = arr.copy()
                        stem = f"{name}_{variant}_conf{conf:.2f}_n{n}"
                        cv2.imwrite(str(OUT_DIR / f"{stem}.jpg"), vis[:, :, ::-1])
                        print(f"{stem}: 检出 {n} 辆" + (f"  conf {row['conf_min']}~{row['conf_max']}" if n else ""))
    print(f"\n结果写入 {csv_path}；叠加图在 {OUT_DIR}")


def _read_variants(src, c0, r0, tile):
    win = rasterio.windows.Window(c0, r0, tile, tile)
    arr = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0)).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enh = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return [("orig", arr), ("clahe", enh)]


def _draw(arr, obb):
    """在 RGB 数组上画 OBB 旋转框（res.obb 提供 .xyxyxyxy 四点）。"""
    vis = arr.copy()
    pts_all = obb.xyxyxyxy.cpu().numpy()  # (N,4,2)
    for pts in pts_all:
        cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 255, 0), 2)
    return vis


if __name__ == "__main__":
    main()
