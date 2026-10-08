"""DeepSeek 视觉模型边界探测（实验用）。

读取 .env 中的 DEEPSEEK_* 配置，逐级测试：
  1. 文本连通性
  2. 基础视觉（能否描述一张卫星图）
  3. 停车场布局判读
  4. 车辆计数（与人工/检测结果对比）

用法:
  ./.venv/Scripts/python.exe ml/scripts/ds_probe.py [测试项]
"""
import base64
import os
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]


def load_env(p: Path):
    cfg = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


cfg = load_env(ROOT / ".env")
client = OpenAI(
    api_key=cfg["DEEPSEEK_API_KEY"],
    base_url=cfg.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)
MODEL = cfg.get("DEEPSEEK_MODEL", "deepseek-flash")


def img_b64(path: Path, max_side: int = None):
    """读图转 base64 data URL；max_side 限制最长边（模拟不同压缩程度）。"""
    from PIL import Image
    import io
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    b = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b}", im.size


def ask(prompt, image_path=None, max_side=None, model=None, max_tokens=512):
    content = [{"type": "text", "text": prompt}]
    size = None
    if image_path:
        url, size = img_b64(Path(image_path), max_side)
        content.append({"type": "image_url", "image_url": {"url": url}})
    r = client.chat.completions.create(
        model=model or MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content, size


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "text"
    if which == "text":
        out, _ = ask("用一句话回答：1+1等于几？")
        print("文本连通 OK:", out)
    elif which == "describe":
        # 用之前保存的停车场叠加图
        p = ROOT / "data/infer/winter_v5_maxdet/lot_01_c3139_r16737_a6.1_n344.jpg"
        out, size = ask("这是一张卫星图像。请描述你看到了什么（地形、建筑、车辆等）。", p, max_side=1024)
        print(f"[输入尺寸 {size}]")
        print(out)
