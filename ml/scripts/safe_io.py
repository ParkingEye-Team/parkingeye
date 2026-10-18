"""安全写入工具：产物先备份、再原子替换；真值/用户文件禁止静默覆盖。

背景：2026-10-18 发生过“重生成工单覆盖用户已填文件”的事故。此后约定：
  - 任何可能被用户编辑的文件（truth_collection_*、exclude_lots.csv）只写新文件名；
  - 自动产物覆盖前先复制到 data/capacity/backup/<tag>_<timestamp>/；
  - 写入使用临时文件 + os.replace，避免中途失败留下半截文件。

用法（脚本内）：
  from safe_io import write_csv_safe, write_json_safe, write_text_safe, run_id
"""
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKUP_ROOT = ROOT / "data/capacity/backup"


def run_id():
    """运行标识：时间戳 + 短 git commit（无 git 时用 nogit）。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "nogit"
    return f"{stamp}_{commit}"


def backup_file(path, tag="auto", backup_dir=None):
    """把已存在的文件复制到备份目录；返回备份路径或 None。"""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    if backup_dir is None:
        backup_dir = BACKUP_ROOT / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / path.name
    # 同一秒内多次备份同名文件时追加序号，绝不覆盖备份
    i = 1
    while dest.exists():
        dest = backup_dir / f"{path.stem}_{i}{path.suffix}"
        i += 1
    shutil.copy2(path, dest)
    return dest


def _atomic_write_text(path, text, encoding):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def write_text_safe(path, text, encoding="utf-8", tag="auto", backup=True,
                    backup_dir=None):
    path = Path(path)
    if backup:
        backup_file(path, tag=tag, backup_dir=backup_dir)
    _atomic_write_text(path, text, encoding)
    return path


def write_json_safe(path, obj, indent=2, tag="auto", backup=True,
                    backup_dir=None):
    return write_text_safe(
        path, json.dumps(obj, ensure_ascii=False, indent=indent),
        encoding="utf-8", tag=tag, backup=backup, backup_dir=backup_dir)


def write_csv_safe(path, rows, fieldnames=None, encoding="utf-8-sig",
                   tag="auto", backup=True, backup_dir=None):
    path = Path(path)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    if backup:
        backup_file(path, tag=tag, backup_dir=backup_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return path


def copy_binary_safe(path, source, tag="auto", backup=True):
    """图片等二进制产物：先备份旧文件，再写入新内容。"""
    import shutil as _shutil
    path = Path(path)
    if backup:
        backup_file(path, tag=tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copy2(source, path)
    return path
