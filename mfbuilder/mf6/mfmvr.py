from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
from flopy.mf6 import ModflowGwfmvr
from shapely.geometry.base import BaseGeometry

from mfbuilder.dto.mvr import MvrConfig, MvrFeature, MvrEndpoint


def _normalize_boundname(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


class _PackageCellsIndex:
    """Индекс ячеек пакета для быстрого поиска ближайших boundname."""

    def __init__(self, grid, package, stress_period: int, boundname: str | list[str] | None):
        self.grid = grid
        self.package = package
        self.stress_period = stress_period
        self.boundnames = {boundname} if isinstance(boundname, str) else set(boundname or [])
        self._gdf = self._build_index()

    def _build_index(self) -> gpd.GeoDataFrame:
        data = self.package.stress_period_data.get_data(key=self.stress_period)
        if data is None:
            raise ValueError(
                f"В пакете {self.package.package_name} нет данных для stress-периода {self.stress_period}"
            )
        names = getattr(data, "dtype", None).names if hasattr(data, "dtype") else None
        cellids = self._extract_cellids_array(data, names)
        bound_arr = self._extract_boundnames_array(data, names)
        record_idx = np.arange(len(cellids))
        if self.boundnames:
            mask = np.isin(bound_arr, list(self.boundnames))
            cellids = cellids[mask]
            bound_arr = bound_arr[mask]
            record_idx = record_idx[mask]

        if len(cellids) == 0:
            raise ValueError(
                f"Не найдены ячейки в пакете {self.package.package_name} "
                f"для boundname='{self.boundnames or '*'}' в stress-периоде {self.stress_period}"
            )

        grid_idx = [self._cell_index(cid) for cid in cellids]
        geoms = self.grid.geo_dataframe.geometry.iloc[grid_idx].to_numpy()
        gdf = gpd.GeoDataFrame(
            {
                "cellid": cellids,
                "record_index": record_idx,
                "boundname": bound_arr,
                "geometry": geoms,
            },
            geometry="geometry",
        )
        return gdf.reset_index(drop=True)

    def _extract_cellid(self, rec, names):
        if names and "cellid" in names:
            return rec["cellid"]
        return rec[0]

    def _extract_boundname(self, rec, names):
        if names and "boundname" in names:
            return rec["boundname"]
        return None

    def _extract_cellids_array(self, data, names):
        if names and "cellid" in names:
            return data["cellid"]
        return np.array([rec[0] for rec in data])

    def _extract_boundnames_array(self, data, names):
        if names and "boundname" in names:
            arr = np.array([_normalize_boundname(x) for x in data["boundname"]], dtype=object)
        else:
            arr = np.array([None] * len(data), dtype=object)
        return arr

    def _cell_index(self, cellid):
        if isinstance(cellid, tuple):
            if len(cellid) == 2:
                return cellid[1]
            if len(cellid) == 3:
                _, i, j = cellid
                return i * self.grid.ncol + j
        return cellid

    def nearest_cell(self, point: BaseGeometry):
        distances = self._gdf.geometry.distance(point)
        nearest_idx = distances.idxmin()
        return int(self._gdf.loc[nearest_idx, "record_index"])


class _LakBoundnameIndex:
    """Индекс озёр пакета LAK по boundname -> номер озера (ifno).

    В отличие от простых пакетов (DRN/RIV/...), у LAK нет
    stress_period_data с cellid, а MVR-идентификатором озера-приёмника
    является номер озера, а не ячейка сетки - поэтому привязка идёт не по
    ближайшей точке, а напрямую по boundname.
    """

    def __init__(self, package):
        data = package.packagedata.get_data()
        self._map = {
            _normalize_boundname(rec["boundname"]): int(rec["ifno"])
            for rec in data
        }

    def resolve(self, boundname: str) -> int:
        if boundname not in self._map:
            raise ValueError(
                f"В пакете LAK не найдено озеро с boundname='{boundname}'"
            )
        return self._map[boundname]


def _is_lak(package) -> bool:
    return getattr(package, "package_type", None) == "lak"


class MF6MvrBuilder:
    """Построитель пакета MVR на основе точек-перетоков."""

    def __init__(self, model, grid, cfg: MvrConfig | None, packages: dict[str, Any]):
        self.model = model
        self.grid = grid
        self.cfg = dict(cfg) if cfg else {}
        self.packages = packages

    def build(self):
        if not self.cfg:
            return None

        period_records: dict[int, list[tuple]] = defaultdict(list)
        package_refs: set[tuple[str, str]] = set()

        for sp, period in self.cfg.items():
            for link in period.data:
                donor_pkg = self._resolve_package(link.from_.pkg)
                acceptor_pkg = self._resolve_package(link.to.pkg)
                package_refs.add(donor_pkg.package_name)
                package_refs.add(acceptor_pkg.package_name)

                if _is_lak(donor_pkg):
                    raise ValueError(
                        "LAK не может быть 'from' в MVR: сток из озера настраивается "
                        "через блок OUTLETS пакета LAK (sources.lak.<period>.outlets), "
                        "а не через MVR."
                    )
                acceptor_is_lak = _is_lak(acceptor_pkg)

                donor_index = _PackageCellsIndex(self.grid, donor_pkg, sp, link.from_.boundname)
                acceptor_index = (
                    _LakBoundnameIndex(acceptor_pkg) if acceptor_is_lak
                    else _PackageCellsIndex(self.grid, acceptor_pkg, sp, link.to.boundname)
                )
                points = self._load_points(link)

                for geom in points.geometry:
                    from_cell = donor_index.nearest_cell(geom)
                    to_cell = (
                        acceptor_index.resolve(_normalize_boundname(link.to.boundname))
                        if acceptor_is_lak
                        else acceptor_index.nearest_cell(geom)
                    )
                    period_records[sp].append(
                        (
                            donor_pkg.package_name,
                            from_cell,
                            acceptor_pkg.package_name,
                            to_cell,
                            "FACTOR",
                            link.factor,
                        )
                    )
        maxmvr = max(len(recs) for recs in period_records.values())
        packages = [(name,) for name in package_refs]

        return ModflowGwfmvr(
            self.model,
            maxmvr=maxmvr,
            maxpackages=len(packages),
            packages=packages,
            print_flows=True,
            budgetcsv_filerecord=f"{self.model.name}.mvr.bud.csv",
            perioddata=period_records,
        )

    def _resolve_package(self, name: str):
        pkg = self.packages.get(name)
        if pkg is None:
            raise ValueError(f"Пакет '{name}' не найден среди созданных источников/стоков")
        return pkg

    def _load_points(self, link: MvrFeature) -> gpd.GeoDataFrame:
        geom = link.geometry
        if isinstance(geom, (str, Path)):
            gdf = gpd.read_file(geom)
        elif isinstance(geom, BaseGeometry):
            gdf = gpd.GeoDataFrame(geometry=[geom])
        elif isinstance(geom, list):
            gdf = gpd.GeoDataFrame(geometry=geom)
        else:
            raise TypeError(f"Неподдерживаемый тип geometry: {type(geom)}")

        if gdf.empty:
            raise ValueError(f"Геометрия для MVR пуста: {geom}")
        return gdf
