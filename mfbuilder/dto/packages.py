from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, ConfigDict, RootModel, model_validator
from shapely.geometry import base as shapely_base


class SourceSinksFeature(BaseModel):
    """Базовый элемент источника/стока."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    geometry: Path | shapely_base.BaseGeometry | list[shapely_base.BaseGeometry]
    exact: bool = Field(default=False, description="Точное совпадение с ячейками сетки")
    layers: list[int] | str | None = Field(default=None, description="Целевые слои (список или поле)")
    # Опции для формирования boundname
    boundname: str | None = Field(default=None, description="Один boundname на весь объект")
    boundname_field: str | None = Field(default=None, description="Имя поля в GeoDataFrame для boundname")
    boundname_prefix: str | None = Field(default=None, description="Префикс для автонумерации boundname")
    # Фильтрация геометрии по атрибутам
    filter: str | None = Field(default=None, description="Условие фильтрации строк GeoDataFrame (pandas query)")

    @classmethod
    def load_geometry(cls, geom):
        """Преобразует путь в геометрию shapely."""
        import geopandas as gpd
        from pathlib import Path
        from shapely.geometry.base import BaseGeometry

        if isinstance(geom, (str, Path)):
            path = Path(geom)
            if not path.exists():
                raise FileNotFoundError(f"Файл геометрии не найден: {path}")
            gdf = gpd.read_file(path)
            if gdf.empty:
                raise ValueError(f"Пустой файл геометрии: {path}")
            return gdf
        if isinstance(geom, BaseGeometry):
            return [geom]
        if isinstance(geom, list):
            return geom
        raise TypeError(f"Некорректный тип geometry: {type(geom)}")

    def get_filtered_geometry(self):
        """Загружает геометрию и применяет фильтр (если задан)."""
        import geopandas as gpd

        result = self.load_geometry(self.geometry)
        if self.filter is None or not isinstance(result, gpd.GeoDataFrame):
            return result
        filtered = result.query(self.filter).reset_index(drop=True)
        if filtered.empty:
            raise ValueError(
                f"После применения фильтра '{self.filter}' не осталось объектов в геометрии."
            )
        return filtered

    def resolve_layers(self, geom_gdf, geom_index: int) -> list[int]:
        """Возвращает список слоёв для текущей фичи и геометрии."""
        import numpy as np

        if self.layers is None:
            return [1]  # слой по умолчанию

        if isinstance(self.layers, list):
            return self.layers

        if isinstance(self.layers, str):
            if self.layers not in geom_gdf.columns:
                raise ValueError(f"В GeoDataFrame нет столбца '{self.layers}' для определения слоя.")

            lay_val = geom_gdf.iloc[geom_index][self.layers]
            # На случай, если значение может быть np.int64 или Series
            if np.isscalar(lay_val):
                return [int(lay_val)]
            return [int(x) for x in np.atleast_1d(lay_val)]

        raise TypeError(f"Некорректный тип layers: {type(self.layers)}")


class RivFeature(SourceSinksFeature):
    stage: float | Path | str
    cond: float | str
    depth: float | str | None = None  # может быть выражением типа "stage - 3"

    def postprocess(self, values: dict[str, float]) -> dict[str, float]:
        if "stage" in values and "depth" in values:
            values["elev"] = values["stage"] - (values["depth"] or 0)
        return values


class GhbFeature(SourceSinksFeature):
    bhead: float | Path | str
    cond: float


class DrnFeature(SourceSinksFeature):
    head: float | Path | str
    cond: float | str


class WelFeature(SourceSinksFeature):
    rate: float | str  # может быть числом или именем поля (например, "rate")


class ChdFeature(SourceSinksFeature):
    """Граница постоянного напора (CHD)."""
    head: float | Path | str


class LakFeature(SourceSinksFeature):
    """Озеро/пруд (полигон), задаётся пакетом LAK."""
    head: float | Path | str  # начальный уровень озера (strt); при status=CONSTANT - принудительный уровень (например, НПУ) для этого периода
    cond: float | str  # проводимость ложа озера (bedleak)
    runoff: float | str = 0.0  # поверхностный приток в озеро (игнорируется, если заданы precip и runoff_coeff - см. ниже)
    evaporation: float | str = 0.0  # испарение с зеркала озера
    status: str | None = None  # ACTIVE (по умолчанию) | INACTIVE | CONSTANT - режим озера на этот stress-период
    # Автоматический расчёт runoff = (площадь_пруда * catchment_multiplier) * precip * runoff_coeff.
    # Заполняются оба (precip и runoff_coeff) - тогда поле runoff выше не используется.
    precip: float | str | None = None  # осадки за период, м/сут
    runoff_coeff: float | str | None = None  # доля осадков, уходящая в поверхностный сток (0..1)
    catchment_multiplier: float | str = 4.0  # во сколько раз водосбор больше зеркала пруда (грубая оценка)
    # Прямые осадки на зеркало озера (laksetting "rainfall", L/T - MF6 сам умножает на площадь).
    # Если precip задан, а auto_rainfall не отключён - rainfall = precip автоматически.
    auto_rainfall: bool = True


class LakOutletFeature(BaseModel):
    """Связь между озёрами (переливные трубы/водосбросы) — блок OUTLETS пакета LAK.

    Каждая линия соединяет два полигона озёр (LakFeature): начало линии -
    исток (lakein), конец линии - приёмник (lakeout). Если конец линии не
    попадает ни в одно озеро, сток считается уходящим из модели (lakeout=0).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    geometry: Path | shapely_base.BaseGeometry | list[shapely_base.BaseGeometry]
    couttype: str = "WEIR"
    invert: float | str  # отметка порога перелива/трубы
    width: float | str = 0.5  # ширина водослива / условный диаметр трубы
    rough: float | str = 0.61  # коэффициент шероховатости (только для MANNING)
    slope: float | str = 9.8  # уклон (только для MANNING)
    match_tolerance: float = 1.0  # допуск (м) для привязки концов линии к полигону озера

    def get_filtered_geometry(self):
        return SourceSinksFeature.load_geometry(self.geometry)


class SourceSinksZone(BaseModel):
    """Одна зона источников (например, riv.0 или wel.0)."""
    data: list[RivFeature | WelFeature | DrnFeature | GhbFeature | LakFeature | ChdFeature]  # или Union позже
    outlets: list[LakOutletFeature] = Field(default_factory=list)  # только для lak


class SourcePackageConfig(BaseModel):
    """Конфигурация одного пакета источников/стоков с доп. флагами."""
    mover: bool = False
    periods: dict[int, SourceSinksZone] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy(cls, value: Any):
        # Поддержка старого вида: {0: {...}, 1: {...}}
        if isinstance(value, dict):
            mover = value.get("mover", False)
            periods = {int(k): v for k, v in value.items() if k != "mover"}
            return {"mover": mover, "periods": periods}
        return value

    def __getitem__(self, key: int) -> SourceSinksZone:
        return self.periods[key]


class SourcesSinksConfig(RootModel[dict[str, SourcePackageConfig]]):
    """Модель для секции sources/sinks (универсальный словарь пакетов с флагом mover)."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_packages(cls, value: Any):
        if isinstance(value, dict):
            return {k: SourcePackageConfig.model_validate(v) for k, v in value.items()}
        return value

    def __getitem__(self, key: str) -> Any:
        return self.root[key]

    def keys(self):
        return self.root.keys()

    def items(self):
        return self.root.items()
