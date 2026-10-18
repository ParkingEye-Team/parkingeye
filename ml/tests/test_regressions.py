"""回归测试：覆盖 2026-10-17 审查（doc/records/audit-2026-10-17.md）中的关键 P0/P1 修复点。

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


class TestPoiCoordinates(unittest.TestCase):
    """POI 坐标转换：GCJ-02 与 WGS84 往返误差应小于 0.2m。"""

    def test_wgs84_gcj02_roundtrip(self):
        mod = load_script("23_fetch_amap_poi.py")
        lng, lat = 103.9940, 30.7196  # 成都北站片区
        glng, glat = mod.wgs84_to_gcj02(lng, lat)
        # 成都 GCJ-02 偏移约 300-500m，转换不能是恒等
        self.assertGreater(abs(glng - lng) + abs(glat - lat), 0.001)
        wlng, wlat = mod.gcj02_to_wgs84(glng, glat)
        self.assertLess(abs(wlng - lng), 2e-6)
        self.assertLess(abs(wlat - lat), 2e-6)

    def test_rank_percentile(self):
        mod = load_script("24_poi_demand_index.py")
        self.assertEqual(mod.rank_percentile([1, 2, 3]), [0.0, 50.0, 100.0])
        self.assertEqual(mod.rank_percentile([5]), [50.0])


class TestCalibrationContract(unittest.TestCase):
    """doc/records/audit-2026-10-18.md P0-1：split 必须被尊重，布局自由文本必须归一化。"""

    def test_layout_prefix_normalization(self):
        mod = load_script("22_capacity_calibrate.py")
        self.assertEqual(mod.normalize_layout("垂直式为主"), "垂直式")
        self.assertEqual(mod.normalize_layout("平行式为主"), "平行式")
        self.assertEqual(mod.normalize_layout("斜列式为主"), "斜列式")
        self.assertEqual(mod.normalize_layout("路边停车"), "平行式")
        self.assertEqual(mod.normalize_layout("混合"), "不规则")
        self.assertEqual(mod.normalize_layout(""), "垂直式")

    def test_split_is_respected(self):
        mod = load_script("22_capacity_calibrate.py")
        rows = [{"lot_id": f"L{i:02d}", "split_raw": "calib" if i < 17 else "test"}
                for i in range(24)]
        calib, test, source = mod.resolve_split(rows)
        self.assertEqual(source, "csv")
        self.assertEqual(len(calib), 17)
        self.assertEqual(len(test), 7)
        self.assertEqual([r["lot_id"] for r in test], [f"L{i:02d}" for i in range(17, 24)])

    def test_partial_split_rejected(self):
        mod = load_script("22_capacity_calibrate.py")
        rows = [{"lot_id": "A", "split_raw": "calib"},
                {"lot_id": "B", "split_raw": ""}]
        with self.assertRaises(SystemExit):
            mod.resolve_split(rows)


class TestEvalChannel(unittest.TestCase):
    """19 号：GE 权重自动 BGR，DOTA 权重自动 RGB。"""

    def test_infer_channel(self):
        mod = load_script("19_eval_three_domains.py")
        self.assertEqual(
            mod.infer_channel("runs/obb/ge_finetune_v3/weights/best.pt"), "bgr")
        self.assertEqual(
            mod.infer_channel("runs/obb/dota_vehicle_obb/weights/best.pt"), "rgb")


class TestSanityDedup(unittest.TestCase):
    """doc/records/audit-2026-10-18.md P1-2：跨窗口检出必须能按距离去重。"""

    def test_dedup_centers(self):
        mod = load_script("21_capacity_sanity.py")
        pts = [(0.0, 0.0), (1.5, 0.0), (5.0, 0.0), (5.5, 0.2)]
        kept = mod.dedup_centers(pts, 2.0)
        self.assertEqual(kept, [(0.0, 0.0), (5.0, 0.0)])


class TestPoiKeyMask(unittest.TestCase):
    """doc/records/audit-2026-10-18.md P1-6：meta/控制台不得暴露 Key 的前 4 位。"""

    def test_mask_only_last4(self):
        mod = load_script("23_fetch_amap_poi.py")
        masked = mod.mask_key("abcd1234efgh5678")
        self.assertEqual(masked, "***5678")
        self.assertNotIn("abcd", masked)


class TestPoiExcluded(unittest.TestCase):
    """doc/records/audit-2026-10-18.md P0-3：24 号必须读取 exclude_lots.csv。"""

    def test_load_excluded(self):
        mod = load_script("24_poi_demand_index.py")
        with tempfile.TemporaryDirectory() as d:
            old = mod.CAP_DIR
            mod.CAP_DIR = Path(d)
            try:
                (Path(d) / "exclude_lots.csv").write_text(
                    "lot_id,reason\nseg_001,测试排除\n", encoding="utf-8")
                self.assertEqual(mod.load_excluded(), {"seg_001"})
            finally:
                mod.CAP_DIR = old

    def test_is_excluded(self):
        mod = load_script("24_poi_demand_index.py")
        self.assertTrue(mod.is_excluded({"id": "seg_001"}, {"seg_001"}))
        self.assertTrue(mod.is_excluded({"id": "seg_002", "excluded": True}, set()))
        self.assertFalse(mod.is_excluded({"id": "seg_003"}, set()))


class TestSafeIo(unittest.TestCase):
    """doc/records/audit-2026-10-18.md P0-2：覆盖产物前必须备份，且原子替换。"""

    def test_write_csv_backs_up(self):
        import csv as _csv
        mod = load_script("safe_io.py")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            p = d / "out.csv"
            p.write_text("a,b\n0,0\n", encoding="utf-8")
            backup_dir = d / "backup"
            mod.write_csv_safe(p, [{"a": 1, "b": 2}], backup_dir=backup_dir, tag="test")
            with open(p, encoding="utf-8-sig") as f:
                rows = list(_csv.DictReader(f))
            self.assertEqual(rows, [{"a": "1", "b": "2"}])
            backups = list(backup_dir.glob("out.csv"))
            self.assertEqual(len(backups), 1)
            self.assertIn("0,0", backups[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()