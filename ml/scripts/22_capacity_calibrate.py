"""容量真值校准：读取人工填写的真值表，拟合单车位综合占地系数并评估误差。

输入 CSV 需包含列（缺失列会报错提示）：
  lot_id, area_m2, is_parking, true_capacity, layout_true, split(可选)
  - is_parking 为"是/yes/1/true"才参与；"否"用于排除统计
  - layout_true 支持"平行式为主/垂直式为主/斜列式为主/混合"等写法，按前缀归一化
  - split 为 calib/test；**若提供则严格按该划分**（calib 拟合，test 评估）
  - 若 split 全为空，才用固定种子 70/30 随机划分，并在报告中注明

输出：
  - 控制台：按布局拟合系数、calib/test 的 MAE / MAPE / 平均偏差
  - data/capacity/calibration_report.json（覆盖前自动备份）
  - data/capacity/calibration_report_<run_id>.json（本次运行快照）

用法:
  ./.venv/Scripts/python.exe ml/scripts/22_capacity_calibrate.py <filled.csv>
  ./.venv/Scripts/python.exe ml/scripts/22_capacity_calibrate.py <filled.csv> --report out.json
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import median

from safe_io import run_id, write_json_safe

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/capacity/calibration_report.json"

# 工程初值（作为回退，不参与拟合时使用）
DEFAULT_COEF = {"垂直式": 28, "斜列式": 24, "平行式": 36, "不规则": 32}
YES = {"是", "yes", "y", "1", "true"}
CALIB_ALIASES = {"calib", "calibration", "校准", "拟合"}
TEST_ALIASES = {"test", "测试", "验证", "holdout"}


def normalize_layout(value):
    """把自由文本布局归一化到四个系数类别；混合/未知回退到垂直式。"""
    s = str(value or "").strip()
    if not s:
        return "垂直式"
    if s in DEFAULT_COEF:
        return s
    if s.startswith("平行") or s.startswith("路边") or s.startswith("路侧"):
        return "平行式"
    if s.startswith("垂直"):
        return "垂直式"
    if s.startswith("斜列") or s.startswith("斜"):
        return "斜列式"
    if s.startswith("不规则") or "混合" in s:
        return "不规则"
    return "垂直式"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("is_parking", "")).strip().lower() not in YES:
                continue
            try:
                area = float(r["area_m2"])
                true = float(r["true_capacity"])
            except (KeyError, ValueError, TypeError):
                continue
            if area <= 0 or true <= 0:
                continue
            rows.append({
                "lot_id": str(r.get("lot_id", "")).strip(),
                "area": area, "true": true,
                "layout": normalize_layout(r.get("layout_true", "")),
                "split_raw": str(r.get("split", "")).strip().lower(),
            })
    return rows


def resolve_split(rows):
    """返回 (calib, test, split_source)。提供 split 时严格使用，否则随机 70/30。"""
    raw = [r["split_raw"] for r in rows]
    if any(raw):
        if not all(raw):
            missing = [r["lot_id"] for r in rows if not r["split_raw"]]
            raise SystemExit(f"split 列部分为空，无法确定划分；缺失 lot_id: {missing}")
        calib, test = [], []
        unknown = []
        for r in rows:
            s = r["split_raw"]
            if s in CALIB_ALIASES:
                calib.append(r)
            elif s in TEST_ALIASES:
                test.append(r)
            else:
                unknown.append(f"{r['lot_id']}={s}")
        if unknown:
            raise SystemExit(f"未知 split 值（只接受 calib/test）: {unknown}")
        if not calib or not test:
            raise SystemExit(f"split 必须同时包含 calib 和 test；当前 calib={len(calib)}, test={len(test)}")
        return calib, test, "csv"

    random.Random(42).shuffle(rows)
    n_test = max(1, round(len(rows) * 0.3)) if len(rows) >= 4 else 0
    test = rows[:n_test]
    calib = rows[n_test:] if n_test else rows
    return calib, test, "random"


def main():
    ap = argparse.ArgumentParser(description="容量真值校准")
    ap.add_argument("filled_csv", help="人工填写的真值 CSV")
    ap.add_argument("--report", default=str(OUT), help="最新报告路径（会备份旧文件）")
    args = ap.parse_args()

    rows = load_rows(args.filled_csv)
    if not rows:
        raise SystemExit("没有可用真值行（is_parking=是 且 true_capacity>0）")

    calib, test, split_source = resolve_split(rows)
    print(f"真值样本 {len(rows)} | calib {len(calib)} | test {len(test)} | 划分来源={split_source}")

    # 按布局拟合系数：median(area/true)
    fits = {}
    for lay in DEFAULT_COEF:
        vals = [r["area"] / r["true"] for r in calib if r["layout"] == lay]
        if len(vals) >= 2:
            fits[lay] = round(median(vals), 2)
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

    rid = run_id()
    report = {
        "run_id": rid,
        "input": str(args.filled_csv),
        "n_total": len(rows), "n_calib": len(calib), "n_test": len(test),
        "split_source": split_source,
        "calib_lot_ids": [r["lot_id"] for r in calib],
        "test_lot_ids": [r["lot_id"] for r in test],
        "fitted_coef_m2_per_space": fits,
        "default_coef": DEFAULT_COEF,
        "layout_used": {r["lot_id"]: r["layout"] for r in rows},
        "calib": evaluate(calib),
        "test": evaluate(test) if test else None,
        "caveat": "test 集只在 calib 拟合完成后评估一次；若 test 参与选系数，MAPE 不再是无偏留出估计。",
    }

    out = Path(args.report)
    write_json_safe(out, report, tag="calibration", backup=True)
    write_json_safe(out.parent / f"calibration_report_{rid}.json", report,
                    tag="calibration", backup=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {out}")
    print(f"-> {out.parent / f'calibration_report_{rid}.json'}")


if __name__ == "__main__":
    main()
