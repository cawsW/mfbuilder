from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from shapely.geometry import base as shapely_base


class HeadObservation(BaseModel):
    """Модель для одной группы наблюдений уровня."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    geometry: Path | shapely_base.BaseGeometry | list[shapely_base.BaseGeometry]
    layers: int | str = Field(..., description="Слой или имя столбца слоя в GeoJSON/shape")
    name: str | None = Field(None, description="Имя поля с названием точки")
    head: str | float | None = Field(None, description="Имя поля или значение наблюдённого уровня")
    time: str | float | None = Field(None, description="Имя поля или значение со временем")
    time_condition: list | None = Field(None)

    @classmethod
    def load_geometry(cls, geom):
        """Преобразует путь в GeoDataFrame."""
        import geopandas as gpd
        from shapely.geometry.base import BaseGeometry

        if isinstance(geom, (str, Path)):
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
