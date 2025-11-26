"""
Pilot point generator for groundwater model calibration.

Requirements (install via pip):
    pip install geopandas shapely fiona

All geometries are expected to be in a projected CRS (e.g. EPSG:3857, EPSG:326xx).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
import math

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree  # в начале файла, рядом с остальными import'ами



@dataclass
class PointDensificationConfig:
    """
    Configuration for densifying pilot points around observation points.

    Attributes
    ----------
    geometries : Sequence[BaseGeometry]
        Sequence of point geometries (observation wells, etc.).
    spacing : float
        Spacing of the denser triangular grid (same units as CRS).
    outer_buffer : float
        Radius of the outer buffer around observation points where we densify.
    inner_exclusion : float, default 0.0
        Radius of the inner buffer around points where we *do not* place
        pilot points (to avoid putting pilot points exactly at observation
        locations).
    """
    geometries: Sequence[BaseGeometry]
    spacing: float
    outer_buffer: float
    inner_exclusion: float = 0.0


@dataclass
class LineDensificationConfig:
    """
    Configuration for densifying pilot points along lines.

    Attributes
    ----------
    geometries : Sequence[BaseGeometry]
        Sequence of line geometries.
    spacing : float
        Spacing of the denser triangular grid.
    buffer : float
        Half-width of the band along lines where we densify.
    """
    geometries: Sequence[BaseGeometry]
    spacing: float
    buffer: float


@dataclass
class PolygonDensificationConfig:
    """
    Configuration for densifying pilot points inside polygons.

    Attributes
    ----------
    geometries : Sequence[BaseGeometry]
        Sequence of polygon geometries.
    spacing : float
        Spacing of the denser triangular grid inside these polygons.
    """
    geometries: Sequence[BaseGeometry]
    spacing: float


def _normalize_geometries(
    geometries: Optional[Iterable[BaseGeometry]],
) -> List[BaseGeometry]:
    """
    Normalize various iterable types to a plain list of shapely geometries.
    """
    if geometries is None:
        return []

    # geopandas GeoSeries / GeoDataFrame support
    if hasattr(geometries, "geometry"):
        return [g for g in geometries.geometry if g is not None]

    return [g for g in geometries if g is not None]


def _triangular_points_in_polygon(
    region: BaseGeometry,
    spacing: float,
) -> List[Point]:
    """
    Generate a triangular (hexagonal-like) grid of points inside a polygon.

    Parameters
    ----------
    region : BaseGeometry
        Polygon or multipolygon where points should be generated.
    spacing : float
        Approximate distance between neighboring points (in CRS units).

    Returns
    -------
    List[Point]
        List of points covering the region with triangular pattern.
    """
    if region.is_empty:
        return []

    minx, miny, maxx, maxy = region.bounds

    # For an equilateral triangle of side "spacing", vertical distance between rows:
    y_step = spacing * math.sqrt(3.0) / 2.0

    ys = np.arange(miny - spacing, maxy + spacing, y_step)
    prepared_region = prep(region)

    points: List[Point] = []

    for row_index, y in enumerate(ys):
        # Shift every second row by spacing / 2 to get triangular pattern
        x_offset = 0.0 if row_index % 2 == 0 else spacing / 2.0
        xs = np.arange(minx - spacing, maxx + spacing, spacing) + x_offset

        for x in xs:
            p = Point(float(x), float(y))
            # covers() includes boundary; contains() would exclude boundary
            if prepared_region.covers(p):
                points.append(p)

    return points


def _add_unique_points(
    existing_points: List[Point],
    existing_keys: Set[Tuple[int, int]],
    new_points: Iterable[Point],
    precision: int = 6,
) -> None:
    """
    Add new points to existing_points, avoiding duplicates by rounding coordinates.

    Parameters
    ----------
    existing_points : List[Point]
        List to extend with unique points.
    existing_keys : Set[Tuple[int, int]]
        Set of hashed rounded coordinates already present.
    new_points : Iterable[Point]
        Candidate points to add.
    precision : int, default 6
        Number of decimal digits used when checking duplicates.
    """
    scale = 10 ** precision

    for p in new_points:
        key = (round(p.x * scale), round(p.y * scale))
        if key not in existing_keys:
            existing_keys.add(key)
            existing_points.append(p)


def _filter_by_min_distance(points: List[Point], min_distance: float) -> List[Point]:
    """
    Remove points that are closer than min_distance to each other.
    Оставляет первую встретившуюся точку в каждом "кластере" близких точек.

    Алгоритм:
        - Простая сетка (hash-grid) с шагом cell_size = min_distance.
        - Для каждой точки проверяются только точки в соседних 3x3 ячейках.
    """
    if min_distance <= 0 or len(points) <= 1:
        return points

    # На всякий случай оставляем только Point
    pts = [p for p in points if isinstance(p, Point)]
    if len(pts) <= 1:
        return pts

    cell_size = float(min_distance)

    # Сетка: (ix, iy) -> список индексов в keep_points
    grid: dict[Tuple[int, int], List[int]] = {}

    keep_points: List[Point] = []

    for p in pts:
        ix = int(math.floor(p.x / cell_size))
        iy = int(math.floor(p.y / cell_size))

        too_close = False

        # Проверяем 3x3 окрестность ячейки
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (ix + dx, iy + dy)
                if cell not in grid:
                    continue

                for kp_idx in grid[cell]:
                    q = keep_points[kp_idx]
                    if p.distance(q) < min_distance:
                        too_close = True
                        break

                if too_close:
                    break
            if too_close:
                break

        if too_close:
            # есть точка ближе min_distance → эту не берём
            continue

        # иначе добавляем
        keep_points.append(p)
        new_idx = len(keep_points) - 1
        grid.setdefault((ix, iy), []).append(new_idx)

    return keep_points


def generate_pilot_points(
    domain_polygon: BaseGeometry,
    base_spacing: float,
    crs: str = None,
    point_cfg: Optional[PointDensificationConfig] = None,
    line_cfg: Optional[LineDensificationConfig] = None,
    polygon_cfg: Optional[PolygonDensificationConfig] = None,
    duplicate_precision: int = 6,
    min_distance: Optional[float] = None,   # <--- НОВОЕ
) -> gpd.GeoDataFrame:

    """
    Generate pilot points within a domain polygon with optional local densification.

    Главное отличие от предыдущей версии:
    базовая сетка НЕ строится в зонах учащения (вокруг точек, вдоль линий, в полигонах),
    а только в оставшейся части домена.
    """
    if domain_polygon.is_empty:
        raise ValueError("Domain polygon is empty.")

    # Нормализуем домен и чуть-чуть лечим топологию
    domain = domain_polygon.buffer(0)
    if domain.is_empty:
        raise ValueError("Domain polygon becomes empty after buffering (topology fix).")

    # ----------------------------------------------------------
    # 1. Сначала сформируем зоны учащения
    # ----------------------------------------------------------
    densify_regions: List[BaseGeometry] = []
    point_region: Optional[BaseGeometry] = None
    line_region: Optional[BaseGeometry] = None
    polygon_region: Optional[BaseGeometry] = None

    # 1.1. Зона учащения вокруг точек
    if point_cfg is not None:
        geoms = _normalize_geometries(point_cfg.geometries)
        if geoms:
            obs_union = unary_union(geoms)
            outer = obs_union.buffer(point_cfg.outer_buffer)

            region = outer
            if point_cfg.inner_exclusion > 0.0:
                exclusion = obs_union.buffer(point_cfg.inner_exclusion)
                region = outer.difference(exclusion)

            region = region.intersection(domain)

            if not region.is_empty:
                point_region = region
                densify_regions.append(region)

    # 1.2. Зона учащения вдоль линий
    if line_cfg is not None:
        geoms = _normalize_geometries(line_cfg.geometries)
        if geoms:
            line_union = unary_union(geoms)
            band = line_union.buffer(line_cfg.buffer)
            region = band.intersection(domain)

            if not region.is_empty:
                line_region = region
                densify_regions.append(region)

    # 1.3. Зона учащения внутри полигонов
    if polygon_cfg is not None:
        geoms = _normalize_geometries(polygon_cfg.geometries)
        if geoms:
            poly_union = unary_union(geoms)
            region = poly_union.intersection(domain)

            if not region.is_empty:
                polygon_region = region
                densify_regions.append(region)

    # ----------------------------------------------------------
    # 2. Считаем объединённую зону учащения
    # ----------------------------------------------------------
    if densify_regions:
        densify_union = unary_union(densify_regions).intersection(domain)
        base_region = domain.difference(densify_union)
    else:
        densify_union = None
        base_region = domain

    # ----------------------------------------------------------
    # 3. Строим базовую сетку ТОЛЬКО вне зон учащения
    # ----------------------------------------------------------
    all_points: List[Point] = []
    keys: Set[Tuple[int, int]] = set()

    if not base_region.is_empty:
        base_points = _triangular_points_in_polygon(base_region, base_spacing)
        _add_unique_points(all_points, keys, base_points, precision=duplicate_precision)

    # ----------------------------------------------------------
    # 4. Строим плотные точки в отдельных зонах
    # ----------------------------------------------------------

    # 4.1. Вокруг точек наблюдений
    if point_region is not None:
        dense_points = _triangular_points_in_polygon(point_region, point_cfg.spacing)
        _add_unique_points(
            all_points,
            keys,
            dense_points,
            precision=duplicate_precision,
        )

    # 4.2. Вдоль линий
    if line_region is not None:
        dense_points = _triangular_points_in_polygon(line_region, line_cfg.spacing)
        _add_unique_points(
            all_points,
            keys,
            dense_points,
            precision=duplicate_precision,
        )

    # 4.3. Внутри полигонов
    if polygon_region is not None:
        dense_points = _triangular_points_in_polygon(
            polygon_region,
            polygon_cfg.spacing,
        )
        _add_unique_points(
            all_points,
            keys,
            dense_points,
            precision=duplicate_precision,
        )

    # ----------------------------------------------------------
    # 5. Финальная зачистка по минимальному расстоянию (опционально)
    # ----------------------------------------------------------
    if min_distance is not None and min_distance > 0:
        all_points = _filter_by_min_distance(all_points, min_distance)

    # ----------------------------------------------------------
    # 6. Собираем GeoDataFrame
    # ----------------------------------------------------------
    gdf = gpd.GeoDataFrame(
        {"name": [f"pp{idx}" for idx in range(len(all_points))]},
        geometry=all_points,
        crs=crs,
    )
    return gdf



def save_pilot_points(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
) -> None:
    """
    Save pilot points to GeoJSON or ESRI Shapefile, based on file extension.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with pilot points.
    path : str or Path
        Output file path. Supported extensions: .geojson, .json, .shp
    """
    output_path = Path(path)
    suffix = output_path.suffix.lower()

    if suffix == ".shp":
        driver = "ESRI Shapefile"
    elif suffix in {".geojson", ".json"}:
        driver = "GeoJSON"
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Use .shp or .geojson/.json."
        )

    gdf.to_file(output_path, driver=driver)
