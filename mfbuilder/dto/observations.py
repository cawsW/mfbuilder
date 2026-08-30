from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from shapely.geometry import base as shapely_base

from mfbuilder.utils.geometry import try_parse_wkt


class HeadObservation(BaseModel):
    """Модель для одной группы наблюдений уровня."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    geometry: Path | shapely_base.BaseGeometry | list[shapely_base.BaseGeometry]
    layers: int | str = Field(..., description="Слой или имя столбца слоя в GeoJSON/shape")
    name: str | None = Field(None, description="Имя поля с названием точки")
    head: str | float | None = Field(None, description="Имя поля или значение наблюдённого уровня")
    time: str = Field(description="Имя поля со временем/периодом наблюдения (например, 'year')")
    time_condition: list | None = Field(None)
    obs_type: Literal["head", "drawdown"] = Field(
        default="head", description="Тип наблюдения MF6 (модельные, через ModflowUtlobs): head | drawdown")
    output: Path | None = Field(
        default=None,
        description="Куда сохранить этот же файл с добавленными колонками head_sim/res после расчёта "
                    "модели; по умолчанию — output/vectors/<имя входного файла>"
    )

    @classmethod
    def load_geometry(cls, geom):
        """Преобразует путь в GeoDataFrame."""
        import geopandas as gpd
        from shapely.geometry.base import BaseGeometry

        if isinstance(geom, (str, Path)):
            wkt_geom = try_parse_wkt(str(geom))
            if wkt_geom is not None:
                return gpd.GeoDataFrame(geometry=[wkt_geom])
            path = Path(geom)
            if not path.exists():
                raise FileNotFoundError(f"Файл не найден: {path}")
            gdf = gpd.read_file(path)
            if gdf.empty:
                raise ValueError(f"Файл {path} пуст.")
            return gdf
        if isinstance(geom, BaseGeometry):
            import geopandas as gpd
            return gpd.GeoDataFrame(geometry=[geom])
        raise TypeError(f"Некорректный тип geometry: {type(geom)}")


class ObservationsConfig(BaseModel):
    heads: list[HeadObservation] | None = None
