from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, ConfigDict, RootModel
from shapely.geometry import base as shapely_base


class SourceSinksFeature(BaseModel):
    """Базовый элемент источника/стока."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    geometry: Path | shapely_base.BaseGeometry | list[shapely_base.BaseGeometry]
    exact: bool = Field(default=False, description="Точное совпадение с ячейками сетки")
    layers: list[int] | str | None = Field(default=None, description="Целевые слои (список или поле)")

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

        # 👇 добавляем метод сюда

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
    cond: float
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
    cond: float


class WelFeature(SourceSinksFeature):
    rate: float | str  # может быть числом или именем поля (например, "rate")


class SourceSinksZone(BaseModel):
    """Одна зона источников (например, riv.0 или wel.0)."""
    data: list[RivFeature | WelFeature | DrnFeature | GhbFeature]  # или Union позже


class SourcesSinksConfig(RootModel[dict[str, dict[int, SourceSinksZone]]]):
    """Модель для секции sources/sinks (универсальный словарь пакетов)."""

    def __getitem__(self, key: str) -> Any:
        return self.root[key]

    def keys(self):
        return self.root.keys()

    def items(self):
        return self.root.items()
