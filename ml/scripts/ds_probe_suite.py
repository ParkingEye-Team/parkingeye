"""DeepSeek 视觉模型边界探测（面向本项目需求）。

测试项：
  T1 停车场布局判读（垂直/平行/斜列）——容量估计的关键输入
  T2 车辆计数（与我们的检测 344 / 人工目测对比）
  T3 分辨率极限（1024→256 递减，看何时失去车辆识别能力）
  T4 停车场鉴别（停车场 vs 非停车场窗口）
  T5 批量成本测量（token 消耗）

结果写入 data/infer/ds_probe_results.csv

用法:
  ./.venv/Scripts/python.exe ml/scripts/ds_probe_suite.py [T1|T2|T3|T4|T5|all]
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_probe import client, MODEL, img_b64, ROOT

TIF = ROOT / "data/imagery/JL1KF01C/JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH" \
      / "JL1KF01C_PMSL2_20231218114103_200220439_101_0022_001_L3D_PSH.tif"
OUT_CSV = ROOT / "data/infer" / "ds_probe_results.csv"
TMP = ROOT / "data/infer" / "ds_probe_imgs"
TMP.mkdir(parents=True, exist_ok=True)

LOTS = [  # (名称, col, row)
    ("lot_01_6.1ha", 3139, 16737),
    ("lot_02_4.8ha", 17241, 26037),
    ("lot_03_3.1ha", 5083, 20588),
]


def crop_window(col, row, size=1024, out_name=None):
    """从冬季大图裁窗口存 PNG（供 API 用）。"""
    with rasterio.open(TIF) as src:
        win = rasterio.windows.Window(col, row, size, size)
        arr = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0)).astype(np.uint8)
    from PIL import Image
    p = TMP / (out_name or f"crop_{col}_{row}.png")
    Image.fromarray(arr).save(p)
    return p


def ask_api(prompt, image_path, max_side=None, max_tokens=4000):
    url, size = img_b64(Path(image_path), max_side)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        max_tokens=max_tokens,
    )
    usage = r.usage
    return r.choices[0].message.content, size, usage


def run(tests):
    rows = []
    if "T1" in tests or "all" in tests:
        print("\n=== T1 布局判读 ===")
        for name, c, r_ in LOTS:
            img = crop_window(c, r_, 1024, f"{name}_raw.png")
            q = ("这是成都的卫星影像（0.5米分辨率），图中是一个露天停车场。"
                 "请判断：1) 该停车场的车位排布是垂直式、平行式还是斜列式（可混合，请给出主要类型和占比估计）；"
                 "2) 车位线是否可见；3) 你的判断置信度（高/中/低）。"
                 "请简洁作答，150字内。")
            ans, size, u = ask_api(q, img)
            rows.append({"test": "T1", "item": name, "answer": ans,
                         "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens})
            print(f"[{name}] {ans[:200]}")
    if "T2" in tests or "all" in tests:
        print("\n=== T2 车辆计数 ===")
        for name, c, r_ in LOTS:
            img = crop_window(c, r_, 1024, f"{name}_raw.png")
            q = ("这是0.5米分辨率的卫星影像。请估计图中停车场内（以及画面中）停放的车辆总数，"
                 "不要数车位线，只数能看到的车辆。给出：1) 你的估计数字；2) 不确定度范围；3) 哪些因素影响计数。"
                 "简洁作答，150字内。")
            ans, size, u = ask_api(q, img)
            rows.append({"test": "T2", "item": name, "answer": ans,
                         "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens})
            print(f"[{name}] {ans[:200]}")
    if "T3" in tests or "all" in tests:
        print("\n=== T3 分辨率极限 ===")
        name, c, r_ = LOTS[0]
        img = crop_window(c, r_, 1024, f"{name}_raw.png")
        q = "图中能看到车辆吗？如果能，描述车辆的数量级和清晰程度；如果不能，说明为什么。60字内。"
        for ms in [1024, 768, 512, 384, 256, 160]:
            ans, size, u = ask_api(q, img, max_side=ms, max_tokens=2000)
            rows.append({"test": "T3", "item": f"max_side={ms}", "answer": ans,
                         "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens})
            print(f"[{ms}px] {ans[:150]}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "item", "answer", "prompt_tokens", "completion_tokens"])
        w.writeheader()
        w.writerows(rows)
    tot_p = sum(r["prompt_tokens"] for r in rows)
    tot_c = sum(r["completion_tokens"] for r in rows)
    print(f"\n共 {len(rows)} 次调用 | prompt {tot_p} + completion {tot_c} tokens -> {OUT_CSV}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    run([which] if which != "all" else ["all"])
