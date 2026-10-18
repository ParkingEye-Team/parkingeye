"""回归测试：覆盖 2026-10-17 审查（doc/11）中的关键 P0/P1 修复点。

运行:
  ./.venv/Scripts/python.exe -m unittest discover -s ml/tests -v
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import box

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# 让脚本内的同级 import（如 osm_utils）可用
sys.path.insert(0, str(SCRIPTS))


def load_script(filename: str):
    """按文件路径加载数字开头的脚本模块（import 语法不支持以数字开头）。"""
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestWindowCoverage(unittest.TestCase):
    """P0-5：滑窗必须覆盖右/下边缘。"""

    def test_tile_positions_include_last_window(self):
        mod = load_script("11_infer_seg_full.py")
        pos = mod.tile_positions(4800, 1024, 824)
        self.assertEqual(pos[-1], 4800 - 1024)
        self.assertIn(3776, pos)
        pos2 = mod.tile_positions(3134, 1024, 824)
        self.assertEqual(pos2[-1], 3134 - 1024)
        self.assertEqual(mod.tile_positions(512, 1024, 824), [0])


class TestWindowDedup(unittest.TestCase):
    """P0-4：预标注窗口去重。"""

    def test_window_iou(self):
        mod = load_script("20_preannot_jl1.py")
        a = (0, 0, 1024)
        self.assertAlmostEqual(mod.window_iou(a, (0, 0, 1024)), 1.0, places=6)
        # 偏移 900px 的窗口 IoU 应低于去重阈值
        self.assertLess(mod.window_iou(a, (900, 0, 1024)), mod.MAX_WINDOW_IOU)
        # 偏移 100px 的窗口应触发去重
        self.assertGreater(mod.window_iou(a, (100, 0, 1024)), mod.MAX_WINDOW_IOU)


class TestCapacityMerge(unittest.TestCase):
    """P0-1：OSM 与分割对同一地块只计一次。"""

    def test_overlapping_lots_merge_to_one(self):
        mod = load_script("18_capacity_engine.py")
        seg = [box(0, 0, 100, 100)]
        osm = [(123, box(10, 0, 110, 100))]
        lots = mod.merge_lots(seg, osm, mod.MATCH_MIN_RATIO, mod.MATCH_MIN_IOU)
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0]["source"], "both")
        # 并集面积 = 110 x 100
        self.assertAlmostEqual(lots[0]["geom"].area, 11000.0, places=3)

    def test_non_overlapping_lots_kept_separate(self):
        mod = load_script("18_capacity_engine.py")
        seg = [box(0, 0, 100, 100)]
        osm = [(456, box(500, 500, 600, 600))]
        lots = mod.merge_lots(seg, osm, mod.MATCH_MIN_RATIO, mod.MATCH_MIN_IOU)
        self.assertEqual(len(lots), 2)
        self.assertEqual({x["source"] for x in lots}, {"seg", "osm"})


class TestAreaSourceRule(unittest.TestCase):
    """OSM 边界过宽时改用分割面积（2026-10-17 规则）。"""

    def test_overseg_osm_uses_seg_geometry(self):
        mod = load_script("18_capacity_engine.py")
        lot = {
            "source": "both", "area_seg": 10000.0, "area_osm": 20000.0,
            "geom": box(0, 0, 200, 100),      # 并集
            "geom_seg": box(0, 0, 100, 100),  # 分割
            "geom_osm": box(0, 0, 200, 100),
        }
        g, src, ratio = mod.choose_geom(lot, 1.5)
        self.assertEqual(src, "seg")
        self.assertAlmostEqual(g.area, 10000.0)
        self.assertAlmostEqual(ratio, 2.0)

    def test_normal_ratio_uses_union(self):
        mod = load_script("18_capacity_engine.py")
        lot = {
            "source": "both", "area_seg": 10000.0, "area_osm": 11000.0,
            "geom": box(0, 0, 110, 100),
            "geom_seg": box(0, 0, 100, 100),
            "geom_osm": box(10, 0, 110, 100),
        }
        g, src, ratio = mod.choose_geom(lot, 1.5)
        self.assertEqual(src, "union")
        self.assertAlmostEqual(g.area, 11000.0)


class TestChannelRoundtrip(unittest.TestCase):
    """P0-3：通道写盘/读回约定。"""

    def test_correct_bgr_write_preserves_rgb(self):
        import cv2
        rgb = np.zeros((4, 4, 3), np.uint8)
        rgb[:, :] = (255, 0, 0)  # 红
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ok.png"
            # 正确路径：RGB -> BGR -> imwrite（文件应是红色）
            cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            back = np.asarray(Image.open(p).convert("RGB"))
            self.assertEqual(tuple(back[0, 0]), (255, 0, 0))

            # 旧 DOTA 路径：直接把 RGB 交给 imwrite，文件会 R/B 交换（蓝色）
            p2 = Path(d) / "swapped.png"
            cv2.imwrite(str(p2), rgb)
            back2 = np.asarray(Image.open(p2).convert("RGB"))
            self.assertEqual(tuple(back2[0, 0]), (0, 0, 255))


class TestTileGeoref(unittest.TestCase):
    """P2-4：普通 PNG 不应被标记为 georeferenced。"""

    def test_plain_png_not_georeferenced(self):
        mod = load_script("01_tile_imagery.py")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "plain.png"
            Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(p)
            _, meta = mod.read_image(p)
            self.assertFalse(meta["georeferenced"])


if __name__ == "__main__":
    unittest.main()