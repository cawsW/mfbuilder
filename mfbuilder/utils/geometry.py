from shapely import wkt
from shapely.geometry.base import BaseGeometry


def try_parse_wkt(value: str) -> BaseGeometry | None:
    """Пытается распарсить строку как WKT-геометрию (например,
    "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"). Возвращает None, если это не
    WKT — тогда вызывающий код должен трактовать значение как путь до файла.
    """
    try:
        return wkt.loads(value)
    except Exception:
        return None
