"""停车场语义分割训练（YOLO11n-seg）——含掩膜→多边形标签自动转换。

09_prep_seg_manual.py 产出的 masks/*.png 是二值掩膜，Ultralytics seg 训练需要
polygon txt。本脚本先自动把掩膜轮廓转成 YOLO polygon 标签（class 0=parking），
再启动训练。

用法:
  python ml/scripts/10_train_seg.py --epochs 120 --batch 8 --imgsz 512
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def mask_to_yolo_txt(mask_path: Path, out_txt: Path, img_size, min_area=30):
    """把单张二值掩膜 PNG 转成 YOLO seg txt（多边形，归一化）。"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    h, w = mask.shape
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        # 简化轮廓，减少点数
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.005 * peri, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= w
        pts[:, 1] /= h
        # 归一化边界保护
        if pts.min() < -0.01 or pts.max() > 1.01:
            continue
        line = "0 " + " ".join(f"{x:.4f} {y:.4f}" for x, y in pts)
        lines.append(line)
    if lines:
        out_txt.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def convert_masks(data_dir: Path, split: str):
    """把 masks/{split}/*.png 转成 labels/{split}/*.txt。返回转换张数。"""
    masks_dir = data_dir / "masks" / split
    labels_dir = data_dir / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for mp in masks_dir.glob("*.png"):
        txt = labels_dir / (mp.stem + ".txt")
        n += mask_to_yolo_txt(mp, txt, img_size=512)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/seg_kf01c_manual"))
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default=0)
    ap.add_argument("--name", default="parking_seg")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "images" / "train").exists():
        raise SystemExit(f"数据目录不完整: {data_dir}，先运行 09_prep_seg_manual.py")

    # 1) 掩膜 -> 多边形标签
    for split in ("train", "val"):
        n = convert_masks(data_dir, split)
        print(f"[{split}] 生成含目标标签 {n} 张")

    # 2) 写 yaml（只预测 parking 类；背景是隐式的）
    yaml_path = data_dir / "seg.yaml"
    yaml_path.write_text(
        f"path: {str(data_dir.resolve()).replace(chr(92), '/')}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n  0: parking\n", encoding="utf-8")

    # 3) 训练
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        exist_ok=True,
        amp=True,
        patience=25,
        workers=4,
        cache=False,
    )


if __name__ == "__main__":
    main()
