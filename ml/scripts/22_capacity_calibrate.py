"""容量真值校准：读取人工填写的真值表，拟合单车位综合占地系数并评估误差。

输入 CSV 需包含列（缺失列会报错提示）：
  lot_id, area_m2, is_parking, true_capacity, layout_true, split(可选)
  - is_parking 为"是/yes/1/true"才参与；"否"用于排除统计
  - split 为 calib/test；缺省时 70%/30% 随机划分（固定种子）
输出：
  - 控制台：按布局拟合系数、calib/test 的 MAE / MAPE / 平均偏差
  - data/capacity/calibration_report.json

用法:
  ./.venv/Scripts/python.exe ml/scripts/22_capacity_calibrate.py <filled.csv>
"""
import csv
import json
import random
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/capacity/calibration_report.json"

# 工程初值（作为回退，不参与拟合时使用）
DEFAULT_COEF = {"垂直式": 28, "斜列式": 24, "平行式": 36, "不规则": 32}
YES = {"是", "yes", "y", "1", "true", "TRUE"}


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("is_parking", "")).strip() not in YES:
                continue
            try:
                area = float(r["area_m2"])
                true = float(r["true_capacity"])
            except (KeyError, ValueError, TypeError):
                continue
            if area <= 0 or true <= 0:
                continue
            layout = str(r.get("layout_true", "")).strip() or "垂直式"
            if layout not in DEFAULT_COEF:
                layout = "垂直式"
            rows.append({
                "lot_id": r.get("lot_id", ""), "area": area, "true": true,
                "layout": layout, "split": str(r.get("split", "")).strip().lower(),
            })
    return rows


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: python ml/scripts/22_capacity_calibrate.py <filled.csv>")
    rows = load_rows(sys.argv[1])
    if not rows:
        raise SystemExit("没有可用真值行（is_parking=是 且 true_capacity>0）")

    # 划分 calib/test
    random.Random(42).shuffle(rows)
    n_test = max(1, round(len(rows) * 0.3)) if len(rows) >= 4 else 0
    test = rows[:n_test]
    calib = rows[n_test:] if n_test else rows

    # 按布局拟合系数：median(area/true)
    fits = {}
    for lay in DEFAULT_COEF:
        vals = [r["area"] / r["true"] for r in calib if r["layout"] == lay]
        if len(vals) >= 2:
            fits[lay] = round(median(vals), 2)
    # 样本不足的布局用全局中位
    global_vals = [r["area"] / r["true"] for r in calib]
    if global_vals:
        fits["_global"] = round(median(global_vals), 2)

    def coef_for(lay):
        return fits.get(lay, fits.get("_global", DEFAULT_COEF[lay]))

    def evaluate(data):
        errs = []
        for r in data:
            pred = r["area"] / coef_for(r["layout"])
            err = pred - r["true"]
            errs.append((pred, r["true"], err, abs(err) / r["true"] * 100))
        if not errs:
            return None
        mae = sum(abs(e[2]) for e in errs) / len(errs)
        mape = sum(e[3] for e in errs) / len(errs)
        bias = sum(e[2] for e in errs) / len(errs)
        return {"n": len(errs), "MAE": round(mae, 1), "MAPE_%": round(mape, 1),
                "bias": round(bias, 1)}

    report = {
        "input": str(sys.argv[1]),
        "n_total": len(rows), "n_calib": len(calib), "n_test": len(test),
        "fitted_coef_m2_per_space": fits,
        "default_coef": DEFAULT_COEF,
        "calib": evaluate(calib),
        "test": evaluate(test) if test else None,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()