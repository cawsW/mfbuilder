from pathlib import Path
from typing import Protocol

import numpy as np
import pyemu

from mfbuilder.dto.pest import (
    ParameterGroupConfig,
    ConstantParameterGroup,
    ZoneParameterGroup,
    PilotPointParameterGroup,
    DirectParameterGroup,
    ZoneSourceConfig,
    GeoStructConfig,
)
from mfbuilder.utils.mfdata import PolygonZoneConfig

_VARIO_CLASSES = {
    "spherical": pyemu.geostats.SphVario,
    "exponential": pyemu.geostats.ExpVario,
    "gaussian": pyemu.geostats.GauVario,
}


class IParameterStrategy(Protocol):
    """Одна стратегия параметризации = один способ вызвать PstFrom.add_parameters
    для одной группы параметров конфига. SOLID Open/Closed: новый par_type
    добавляется реализацией этого протокола и регистрацией в
    ParameterStrategyFactory, без изменения существующих стратегий/Director'а."""

    def add(self, pf: pyemu.utils.PstFrom, group: ParameterGroupConfig, grid, calib_ws: Path) -> None: ...


def _base_kwargs(group: ParameterGroupConfig) -> dict:
    """Аргументы PstFrom.add_parameters, общие для всех типов параметризации."""
    return dict(
        filenames=group.files,
        par_name_base=group.par_name_base,
        pargp=group.pargp or group.par_name_base,
        index_cols=group.index_cols,
        use_cols=group.use_cols,
        use_rows=group.use_rows,
        initial_value=group.initial_value,
        lower_bound=group.lower_bound,
        upper_bound=group.upper_bound,
        ult_lbound=group.ult_lbound,
        ult_ubound=group.ult_ubound,
        transform=group.transform,
        **group.extra,
    )


class ConstantParameterStrategy:
    """Один общий множитель на все files (целиком либо на use_rows/use_cols)."""

    def add(self, pf, group: ConstantParameterGroup, grid, calib_ws: Path) -> None:
        pf.add_parameters(par_type="constant", **_base_kwargs(group))


class ZoneParameterStrategy:
    """Свой множитель на каждую зону. Источник зон переиспользует тот же
    PolygonZoneConfig (mfbuilder.utils.mfdata), что и зональность параметров
    самой модели (NPF/RCH/...) — один и тот же способ 'полигон + поле ->
    массив на сетке' что при сборке модели, что при настройке PEST.
    Если файл зон не задан — зоны строятся из уникальных значений уже
    посчитанного массива (например, когда K33 в модели изначально задан
    двумя-тремя дискретными значениями по литологии)."""

    def add(self, pf, group: ZoneParameterGroup, grid, calib_ws: Path) -> None:
        zone_array = self._zone_array(group.zones, grid, calib_ws, group.files[0])
        pf.add_parameters(par_type="zone", zone_array=zone_array, **_base_kwargs(group))

    @staticmethod
    def _zone_array(zones: ZoneSourceConfig, grid, calib_ws: Path, sample_file: str) -> np.ndarray:
        if zones.file is not None:
            zone_cfg = PolygonZoneConfig(file=zones.file, field=zones.field, default=zones.default)
            return zone_cfg.rasterize(grid).astype(int).reshape(-1)

        arr = np.loadtxt(calib_ws / sample_file).reshape(-1)
        uniques = {value: idx + 1 for idx, value in enumerate(sorted(set(arr.tolist())))}
        return np.array([uniques[v] for v in arr], dtype=int)


class PilotPointParameterStrategy:
    """Пилотные точки + geostruct. Точки берутся из уже готового файла
    (name/x/y или geometry) — генерация сетки точек не выполняется: ни в одном
    реальном проекте пользователя пилотные точки не генерировались программно,
    всегда заранее подготовленный shapefile."""

    def add(self, pf, group: PilotPointParameterGroup, grid, calib_ws: Path) -> None:
        pp_space = self._pp_space(group)
        geostruct = self.geostruct(group.geostruct)
        # pp_space/use_pp_zones как отдельные аргументы add_parameters устарели
        # в текущей версии pyemu — нужен pp_options.
        pf.add_parameters(
            par_type="pilotpoints",
            pp_options={"pp_space": pp_space, "use_pp_zones": False},
            geostruct=geostruct,
            **_base_kwargs(group),
        )

    @staticmethod
    def _pp_space(group: PilotPointParameterGroup):
        import geopandas as gpd

        gdf = gpd.read_file(group.points)
        if "x" not in gdf.columns:
            gdf["x"] = gdf.geometry.x
        if "y" not in gdf.columns:
            gdf["y"] = gdf.geometry.y
        if "name" not in gdf.columns:
            gdf["name"] = [f"{group.par_name_base}{i}" for i in range(len(gdf))]

        if group.zones is not None and group.zones.file is not None:
            zone_gdf = gpd.read_file(group.zones.file)
            zone_geom = zone_gdf.union_all()
            gdf = gdf[gdf.geometry.within(zone_geom)]
            if gdf.empty:
                raise ValueError(
                    f"Ни одна пилотная точка из {group.points} не попала в зону {group.zones.file}."
                )

        return gdf[["name", "x", "y"]].reset_index(drop=True)

    @staticmethod
    def geostruct(cfg: GeoStructConfig) -> pyemu.geostats.GeoStruct:
        vario_cls = _VARIO_CLASSES[cfg.variogram.type]
        vario = vario_cls(
            contribution=cfg.variogram.contribution,
            a=cfg.variogram.range,
            anisotropy=cfg.variogram.anisotropy,
            bearing=cfg.variogram.bearing,
        )
        return pyemu.geostats.GeoStruct(nugget=cfg.nugget, variograms=vario, transform=cfg.transform)


class DirectParameterStrategy:
    """par_style='direct' — PEST пишет физическое значение прямо в файл, а не
    множитель поверх текущего. initial_value здесь — сама стартовая величина,
    а не 1.0. Практический смысл — коэффициент, который затем через
    pre-command хук (PestConfig.hooks) разворачивается в N файлов модели
    (например, амплитуда сезонного питания на все стресс-периоды)."""

    def add(self, pf, group: DirectParameterGroup, grid, calib_ws: Path) -> None:
        pf.add_parameters(par_type="constant", par_style="direct", **_base_kwargs(group))


class ParameterStrategyFactory:
    """Реестр стратегий параметризации, ключ — ParameterGroupConfig.par_type.

    SOLID Open/Closed: новый тип параметризации добавляется register()'ом
    извне, без правки фабрики или PestDirector.
    """

    def __init__(self) -> None:
        self._map: dict[str, IParameterStrategy] = {
            "constant": ConstantParameterStrategy(),
            "zone": ZoneParameterStrategy(),
            "pilotpoints": PilotPointParameterStrategy(),
            "direct": DirectParameterStrategy(),
        }

    def register(self, par_type: str, strategy: IParameterStrategy) -> None:
        self._map[par_type] = strategy

    def get(self, par_type: str) -> IParameterStrategy:
        strategy = self._map.get(par_type)
        if strategy is None:
            raise ValueError(f"Неизвестный тип параметризации: {par_type}")
        return strategy
