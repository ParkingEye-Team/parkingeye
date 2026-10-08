"""GE 成都截图标注 JSON -> YOLO-OBB 数据集（车辆检测成都域微调）。

输入：data/ge_annot/raw/<图名>.json（X-AnyLabeling 导出，与图同目录，旋转框）
      每个 shape: {label:'vehicle', points: 4 角点 [[x,y]...]（像素，绕中心逆时针或任意序）}
输出：data/ge_yolo/train|val/images/*.jpg + labels/*.txt（YOLO-OBB 归一化格式）
      附带 train/val 划分（≥5 张按 80/20，<5 张全 train）与 data/ge_yolo/ge_obb.yaml

YOLO-OBB 文本行: cls x1 y1 x2 y2 x3 y3 x4 y4  （归一化 0-1，四点按序构成凸四边形）

用法:
  ./.venv/Scripts/python.exe ml/scripts/14_ge_obb_to_yolo.py
"""
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/ge_annot" / "raw"   # X-AnyLabeling 标注 JSON 与图同目录
OUT_DIR = ROOT / "data" / "ge_yolo"
CLASSES = ["vehicle"]  # 单类，与 DOTA 微调一致
VAL_FRAC = 0.2


def main():
    # 标注 JSON 由 X-AnyLabeling 存在 raw/（与图同目录，便于软件内查看）；脚本从这里读
    jsons = sorted(glob.glob(str(RAW_DIR / "*.json")))
    if not jsons:
        raise SystemExit(f"未找到标注 json: {RAW_DIR}/*.json（先在 X-AnyLabeling 里标完并保存）")

    # 清理重建：产物在 train/ 与 val/ 下，整树删掉避免旧样本残留
    for split in ("train", "val"):
        d = OUT_DIR / split
        if d.exists():
            shutil.rmtree(d)
    for sub in ("images", "labels"):
        (OUT_DIR / "train" / sub).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "val" / sub).mkdir(parents=True, exist_ok=True)

    n_img = n_box = n_skip = 0
    records = []
    for jf in jsons:
        jf = Path(jf)
        # 图名: <场景_序号>，找 raw 下同名 jpg/png
        img_src = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = RAW_DIR / (jf.stem + ext)
            if cand.exists():
                img_src = cand
                break
        if img_src is None:
            print(f"[跳过] {jf.name}: raw 下无同名图片")
            n_skip += 1
            continue

        data = json.loads(jf.read_text(encoding="utf-8"))
        # 用 JSON 里的 imageWidth/imageHeight（若缺失则回退到图片实际尺寸）
        iw = data.get("imageWidth") or data.get("image_width")
        ih = data.get("imageHeight") or data.get("image_height")
        if not (iw and ih):
            from PIL import Image
            im = Image.open(img_src)
            iw, ih = im.size

        lines = []
        for shp in data.get("shapes", []):
            if shp.get("label") not in CLASSES:
                continue
            pts = np.asarray(shp.get("points", []), dtype=np.float64)
            if pts.shape != (4, 2):
                print(f"  [警告] {jf.name}: 非 4 角点 shape 跳过（{pts.shape}）")
                continue
            # 逆时针排序凸四边形（YOLO-OBB 要求顺序：可用极角绕质心排）
            cx, cy = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            pts = pts[np.argsort(ang)]
            # 归一化
            norm = pts / np.array([iw, ih])
            # 越界保护
            if norm.min() < -0.01 or norm.max() > 1.01:
                print(f"  [警告] {jf.name}: 点越界跳过")
                continue
            norm = np.clip(norm, 0, 1)
            line = "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
            lines.append(line)
            n_box += 1

        if not lines:
            print(f"[跳过] {jf.name}: 无有效 vehicle 标注")
            n_skip += 1
            continue
        n_img += 1
        records.append((jf.stem, img_src, lines))

    if not records:
        raise SystemExit("没有可用的标注图片，终止")

    # 划分 train/val：图太少时全部进 train（否则唯一一张会进 val，train 空导致微调失败）
    records.sort(key=lambda r: r[0])
    n_val = 0 if len(records) < 5 else max(1, round(len(records) * VAL_FRAC))
    if n_val == 0:
        print(f"[提示] 标注图仅 {len(records)} 张（<5），暂不切 val，全部作为 train；后续图多了重跑本脚本自动划分。")
    for i, (stem, img_src, lines) in enumerate(records):
        split = "val" if i < n_val else "train"
        # 复制图片（jpg 统一）
        out_img = OUT_DIR / split / "images" / f"{stem}.jpg"
        if img_src.suffix.lower() == ".jpg":
            shutil.copy(img_src, out_img)
        else:
            from PIL import Image
            Image.open(img_src).convert("RGB").save(out_img, "JPEG", quality=95)
        (OUT_DIR / split / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    # yaml
    yaml_path = OUT_DIR / "ge_obb.yaml"
    yaml_path.write_text(
        yaml.safe_dump({
            "path": str(OUT_DIR),
            "train": "train/images",
            "val": "val/images",
            "names": {i: c for i, c in enumerate(CLASSES)},
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    n_train = len(records) - n_val
    print(f"完成：{n_img} 张图 / {n_box} 个车辆框  -> {OUT_DIR}")
    print(f"  train {n_train} 张, val {n_val} 张")
    print(f"  yaml: {yaml_path}")


if __name__ == "__main__":
    main()
