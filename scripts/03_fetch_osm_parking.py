"""拉取指定城市范围的 OSM 露天停车场多边形（Overpass API）。

产出 JSON 可直接作为 scripts/02_rasterize_osm_labels.py 的输入。

用法:
  python scripts/03_fetch_osm_parking.py --bbox 30.57 103.98 30.72 104.18 \
      --output data/osm/chengdu_parking_raw.json
"""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

QUERY = """[out:json][timeout:100];
(
  way["amenity"="parking"]({s},{w},{n},{e});
  relation["amenity"="parking"]({s},{w},{n},{e});
);
out body geom;"""

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("S", "W", "N", "E"), help="南 西 北 东（经纬度）")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    query = QUERY.format(s=args.bbox[0], w=args.bbox[1], n=args.bbox[2], e=args.bbox[3])
    data = urllib.parse.urlencode({"data": query}).encode()

    last_err = None
    for ep in ENDPOINTS:
        try:
            print(f"请求 {ep} ...")
            with urllib.request.urlopen(urllib.request.Request(ep, data=data),
                                        timeout=120) as resp:
                result = json.load(resp)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            n_ways = sum(1 for e in result["elements"] if e["type"] == "way")
            print(f"完成: {n_ways} 个停车场多边形 -> {args.output}")
            return
        except Exception as e:  # noqa: BLE001 换下一个镜像
            last_err = e
            print(f"  失败: {e}")
    raise SystemExit(f"所有 Overpass 镜像均失败: {last_err}")


if __name__ == "__main__":
    main()
