"""Наблюдения уровня для MF6: сборка OBS-пакета и выгрузка результатов.

Два разных этапа жизненного цикла — две разных ответственности (SRP):
  - MF6ObservationsBuilder     — до расчёта: строит ModflowUtlobs.
  - MF6ObservationResultsExporter — после расчёта: дополняет тот же
    geometry-файл наблюдений колонками head_sim/res.

Обе используют один и тот же HeadObservationResolver, чтобы имена/группировка
точек не могли разъехаться между сборкой и выгрузкой (раньше сравнение
результатов вручную парсило сгенерированные имена отдельным кодом).
"""
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from flopy.mf6 import ModflowUtlobs

from mfbuilder.dto.observations import HeadObservation

# obs_type из конфига -> ключевое слово MF6 для ModflowUtlobs.continuous.
# Оба — "модельные" наблюдения (id = (layer-1, cellid)), поэтому строятся
# одинаково; новый ненативный тип (например, привязанный к конкретному
# ГУ-пакету, а не к модели) потребует своего builder'а — сюда его не втиснуть,
# и не нужно: расширение случится в отдельном классе, этот файл трогать не придётся.
MODEL_LEVEL_OBS_TYPES: dict[str, str] = {
    "head": "HEAD",
    "drawdown": "DRAWDOWN",
}


class PointToCellMapper:
    """Ближайшая ячейка сетки для каждой точки наблюдения."""

    def __init__(self, grid):
        if not hasattr(grid, "geo_dataframe"):
            raise RuntimeError("Grid не поддерживает geo_dataframe для маппинга точек наблюдения.")
        self._grid_gdf = grid.geo_dataframe

    def map(self, points: gpd.GeoSeries) -> list:
        from shapely.strtree import STRtree

        tree = STRtree(self._grid_gdf.geometry.values)
        return [self._grid_gdf.index[tree.nearest(p)] for p in points]


class HeadObservationResolver:
    """HeadObservation -> GeoDataFrame с точками, привязанными к сетке.

    К исходным атрибутам/geometry добавляются служебные колонки:
      obs_name   — имя точки в PEST/MF6 (obs_def.name, либо сгенерированное);
      obs_layer  — номер слоя (1-based, как в конфиге);
      obs_cellid — ячейка сетки (row,col) или icpl;
      obs_val    — наблюдённое значение (может быть NaN).

    Единая точка, где решается "какая точка в какую ячейку/слой попадает и
    как называется" — и MF6ObservationsBuilder (сборка пакета), и
    MF6ObservationResultsExporter (выгрузка результата) вызывают именно её,
    поэтому имена точек не могут разойтись между сборкой и выгрузкой.
    """

    def __init__(self, grid):
        self._cell_mapper = PointToCellMapper(grid)

    def resolve(self, obs_def: HeadObservation, name_prefix: str = "obs") -> gpd.GeoDataFrame:
        gdf = obs_def.load_geometry(obs_def.geometry)

        if isinstance(obs_def.head, str):
            gdf = gdf[~gdf[obs_def.head].isna()]
        if obs_def.time not in gdf.columns:
            raise ValueError(f"В GeoDataFrame нет столбца '{obs_def.time}' (HeadObservation.time).")
        if obs_def.time_condition:
            gdf = gdf[gdf[obs_def.time].isin(obs_def.time_condition)]
        if gdf.empty:
            return gdf

        groups = []
        for time_value, gdf_group in gdf.groupby(obs_def.time):
            if gdf_group.empty:
                continue
            gdf_group = gdf_group.copy()
            gdf_group["obs_name"] = self._names(obs_def, gdf_group, time_value, name_prefix)
            gdf_group["obs_layer"] = self._layers(obs_def, gdf_group)
            gdf_group["obs_val"] = self._obsvals(obs_def, gdf_group)
            groups.append(gdf_group)

        if not groups:
            return gdf.iloc[0:0]

        result = pd.concat(groups)
        result["obs_cellid"] = self._cell_mapper.map(result.geometry)
        return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)

    @staticmethod
    def _names(obs_def: HeadObservation, gdf_group: gpd.GeoDataFrame, time_value: Any, prefix: str) -> list[str]:
        if obs_def.name and obs_def.name in gdf_group.columns:
            return gdf_group[obs_def.name].astype(str).tolist()
        return [f"{prefix}_{time_value}_{i}" for i in range(len(gdf_group))]

    @staticmethod
    def _layers(obs_def: HeadObservation, gdf_group: gpd.GeoDataFrame) -> list[int]:
        if isinstance(obs_def.layers, int):
            return [obs_def.layers] * len(gdf_group)
        if isinstance(obs_def.layers, str) and obs_def.layers in gdf_group.columns:
            return gdf_group[obs_def.layers].astype(int).tolist()
        raise ValueError(f"Неверное значение layers: {obs_def.layers}")

    @staticmethod
    def _obsvals(obs_def: HeadObservation, gdf_group: gpd.GeoDataFrame) -> list[float]:
        if isinstance(obs_def.head, (int, float)):
            return [float(obs_def.head)] * len(gdf_group)
        if isinstance(obs_def.head, str) and obs_def.head in gdf_group.columns:
            return gdf_group[obs_def.head].astype(float).tolist()
        return [np.nan] * len(gdf_group)


class MF6ObservationsBuilder:
    """Строит один ModflowUtlobs-пакет из всех HeadObservation конфига."""

    def __init__(self, model, grid, cfg, pname: str = "head_obs", resolver: HeadObservationResolver | None = None):
        self.model = model
        self.grid = grid
        self.cfg = cfg.observations
        self.pname = pname
        self._resolver = resolver or HeadObservationResolver(grid)
        self._built = False

    def build(self):
        if self._built:
            return None
        self._built = True

        if not self.cfg or not self.cfg.heads:
            return None

        records_by_file: dict[str, list[tuple]] = {}
        for idx, obs_def in enumerate(self.cfg.heads):
            mf6_obstype = MODEL_LEVEL_OBS_TYPES.get(obs_def.obs_type)
            if mf6_obstype is None:
                raise ValueError(
                    f"Неизвестный obs_type='{obs_def.obs_type}'. Поддерживаются: {sorted(MODEL_LEVEL_OBS_TYPES)}"
                )

            gdf = self._resolver.resolve(obs_def, name_prefix=f"obs{idx}")
            if gdf.empty:
                continue

            for time_value, gdf_time in gdf.groupby(obs_def.time):
                if gdf_time.empty:
                    continue
                output_filename = f"{self.model.name}.{obs_def.obs_type}_obs_{time_value}.csv"
                records = records_by_file.setdefault(output_filename, [])
                for row in gdf_time.itertuples():
                    cellid = row.obs_cellid
                    cid = (row.obs_layer - 1, *cellid) if isinstance(cellid, tuple) else (row.obs_layer - 1, cellid)
                    records.append((row.obs_name, mf6_obstype, cid))

        if not records_by_file:
            return None

        ModflowUtlobs(
            self.model,
            pname=self.pname,
            filename=f"{self.model.name}.{self.pname}.obs",
            print_input=True,
            digits=10,
            continuous=records_by_file,
        )
        return list(records_by_file.keys())


class MF6ObservationResultsExporter:
    """После расчёта модели сохраняет тот же geometry-файл наблюдений с
    добавленными колонками head_sim (модельный уровень на последний
    рассчитанный момент времени) и res (head - head_sim).

    Источник посчитанных значений — CSV-файлы, зарегистрированные в самом
    OBS-пакете (obs_pkg.continuous.get_data()), а не угаданные по маске имени:
    так выгрузка не может разойтись с тем, что реально построил
    MF6ObservationsBuilder.
    """

    def __init__(self, model, grid, cfg, pname: str = "head_obs", resolver: HeadObservationResolver | None = None):
        self.model = model
        self.cfg = cfg.observations
        self.pname = pname
        self._resolver = resolver or HeadObservationResolver(grid)

    def export(self) -> list[Path]:
        if not self.cfg or not self.cfg.heads:
            return []

        sim_values = self._read_simulated_values()
        if not sim_values:
            return []

        exported = []
        for idx, obs_def in enumerate(self.cfg.heads):
            path = self._export_one(obs_def, sim_values, name_prefix=f"obs{idx}")
            if path is not None:
                exported.append(path)
        return exported

    def _read_simulated_values(self) -> dict[str, float]:
        obs_pkg = self.model.get_package(self.pname)
        if obs_pkg is None:
            return {}

        workspace = Path(self.model.simulation_data.mfpath.get_sim_path())
        values: dict[str, float] = {}
        for filename in obs_pkg.continuous.get_data().keys():
            csv_path = workspace / filename
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            last_row = df.iloc[-1]
            for col in df.columns:
                if col.strip().lower() == "time":
                    continue
                values[col.strip().lower()] = float(last_row[col])
        return values

    def _export_one(self, obs_def: HeadObservation, sim_values: dict[str, float], name_prefix: str) -> Path | None:
        if not isinstance(obs_def.geometry, (str, Path)):
            return None  # геометрия задана "в коде", а не файлом — сохранять некуда

        gdf = self._resolver.resolve(obs_def, name_prefix=name_prefix)
        if gdf.empty:
            return None

        gdf = gdf.drop(columns=["obs_layer", "obs_cellid"])
        gdf["head_sim"] = gdf["obs_name"].str.lower().map(sim_values)
        gdf["res"] = gdf["obs_val"] - gdf["head_sim"]
        gdf = gdf.drop(columns=["obs_name", "obs_val"])

        out_path = self._resolve_output_path(obs_def)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path)
        return out_path

    @staticmethod
    def _resolve_output_path(obs_def: HeadObservation) -> Path:
        if obs_def.output is not None:
            return Path(obs_def.output)
        return Path("../output/vectors") / Path(obs_def.geometry).name
