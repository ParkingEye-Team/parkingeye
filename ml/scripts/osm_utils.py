"""OSM 标签公共工具：露天停车场判定。

统一口径（doc/11 P1-4）：以下 parking 值视为非露天，不作为露天停车资源：
  multi-storey / underground / rooftop / sheds
无 parking 标签的元素默认按露天处理（需在文档中说明）。
"""

NON_SURFACE = {"multi-storey", "underground", "rooftop", "sheds"}


def is_surface_parking(tags: dict) -> bool:
    """判断 OSM 元素是否为露天停车场（默认放行无 parking 标签的元素）。"""
    return str(tags.get("parking", "")).lower() not in NON_SURFACE
