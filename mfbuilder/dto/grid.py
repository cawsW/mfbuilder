import math
from pathlib import Path

from typing import Literal
from typing_extensions import Annotated
from shapely.geometry import Polygon, MultiPolygon, base as shapely_base
import geopandas as gpd

try:
    from pydantic import (BaseModel, ConfigDict, Field, ValidationError, model_validator,
                          computed_field, field_validator)
    from pydantic.types import DirectoryPath, FilePath
except Exception as e:
    raise RuntimeError("pydantic is required: pip install pydantic") from e

from mfbuilder.dto.types import Scalar, DelArray, GridType


class BotmLayer(BaseModel):
    """Один слой из списка botm (допускается один ключ)."""
    elev: FilePath | Scalar | None = None
    thick_top: FilePath | Scalar | None = None
    thick_bot: FilePath | Scalar | None = None

    @model_validator(mode="after")
    def check_one_key(self):
        keys_present = [k for k in ("elev", "thick_top", "thick_bot") if getattr(self, k) is not None]

        if len(keys_present) == 0:
            raise ValueError("Слой botm должен содержать хотя бы один ключ: 'elev', 'thick_top' или 'thick_bot'.")
        if len(keys_present) > 1:
            raise ValueError("Каждый элемент botm должен содержать только один ключ (elev / thick_top / thick_bot).")

        return self


class BaseGridConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid'
    )
    type: GridType
    nlay: Annotated[int, Field(gt=0, default=1, description="Количество слоев")]
    cell_size: Annotated[Scalar, Field(gt=0, description="Размер ячейки сетки")]
    border: Annotated[FilePath | Polygon | None, Field(default=None, description="Граница модели")]
    epsg: Annotated[str | None, Field(default=None, description="Система координат")]
    top: Annotated[FilePath | Scalar | None, Field(default=None, description="Поверхность модели")]
    botm: Annotated[
        list[BotmLayer] | None,
        Field(default=None, description="Список поверхностей (elev / thick_top / thick_bot)")
    ]
    min_thickness: Annotated[float, Field(default=0.0, ge=0.0, description="Минимальная мощность слоя")]
    xmin: Annotated[Scalar | None, Field(default=None, description="Координата x нижнего левого угла сетки")]
    xmax: Annotated[Scalar | None, Field(default=None, description="Координата x нижнего правого края сетки")]
    ymin: Annotated[Scalar | None, Field(default=None, description="Координата y нижнего левого угла сетки")]
    ymax: Annotated[Scalar | None, Field(default=None, description="Координата y верхнего левого угла сетки")]
    nrow: Annotated[int | None, Field(default=None, description="Количество строк сетки")]
    ncol: Annotated[int | None, Field(default=None, description="Количество столбцов сетки")]
    delr: Annotated[DelArray | None, Field(default=None, description="Интервалы вдоль строк")]
    delc: Annotated[DelArray | None, Field(default=None, description="Интервалы вдоль столбцов")]

    @model_validator(mode="after")
    def _post(self):
        self._load_border_if_path()
        self._check_border_and_epsg()
        self._fill_bbox_from_border_if_needed()
        self._infer_counts_and_steps()
        self._final_consistency_checks()
        return self

    @model_validator(mode="after")
    def _check_botm_length(self):
        if self.botm and len(self.botm) != self.nlay:
            raise ValueError(f"Количество элементов botm ({len(self.botm)}) != nlay ({self.nlay})")
        return self

    def _load_border_if_path(self) -> None:
        """Если border — это путь, читаем файл и заменяем на геометрию shapely."""
        if isinstance(self.border, (Polygon, MultiPolygon)) or self.border is None:
            return

        path = Path(self.border)
        if not path.exists():
            raise ValueError(f"Указанный файл границы не найден: {path}")

        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"Векторный файл пуст: {path}")

        geom = gdf.union_all()

        if isinstance(geom, shapely_base.BaseGeometry):
            self.border = geom
        else:
            raise ValueError(f"Не удалось интерпретировать геометрию из {path}")

        # Если epsg не задан, пытаемся взять из CRS GeoDataFrame
        if self.epsg is None and gdf.crs is not None:
            self.epsg = str(gdf.crs.to_epsg() or gdf.crs)

    def _check_border_and_epsg(self) -> None:
        if self.border is not None and self.epsg is None:
            raise ValueError("Если указан border, поле epsg не может быть None.")

    def _fill_bbox_from_border_if_needed(self) -> None:
        if self.border is None:
            return

        xmin, ymin, xmax, ymax = self._bounds_from_border(self.border)
        self.xmin = xmin if self.xmin is None else self.xmin
        self.ymin = ymin if self.ymin is None else self.ymin
        self.xmax = xmax if self.xmax is None else self.xmax
        self.ymax = ymax if self.ymax is None else self.ymax

        for name in ("xmin", "ymin", "xmax", "ymax"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} не определён; не удалось вычислить границы.")

    def _infer_counts_and_steps(self) -> None:
        if self.border is not None:
            dx = float(self.xmax) - float(self.xmin)
            dy = float(self.ymax) - float(self.ymin)
            if self.ncol is None:
                self.ncol = max(1, math.floor(dx / float(self.cell_size)))
            if self.nrow is None:
                self.nrow = max(1, math.floor(dy / float(self.cell_size)))
            if self.delr is None:
                self.delr = [float(self.cell_size)] * int(self.ncol)
            if self.delc is None:
                self.delc = [float(self.cell_size)] * int(self.nrow)
        else:
            if self.xmin is None or self.ymin is None:
                raise ValueError("Когда border не задан, требуются xmin и ymin.")
            have_counts = self.nrow is not None and self.ncol is not None
            have_steps = self.delr is not None and self.delc is not None
            if not (have_counts or have_steps):
                raise ValueError("Когда border не задан, укажите либо nrow+ncol, либо delr+delc.")
            if have_counts and not have_steps:
                self.delr = [float(self.cell_size)] * int(self.ncol)
                self.delc = [float(self.cell_size)] * int(self.nrow)
            if have_steps and not have_counts:
                self.ncol = int(len(self.delr))
                self.nrow = int(len(self.delc))
            self.xmax = self.xmin + self.cell_size * self.ncol
            self.ymax = self.ymin + self.cell_size * self.nrow

    def _final_consistency_checks(self) -> None:
        assert self.nrow and self.ncol
        if not (self.nrow > 0 and self.ncol > 0):
            raise ValueError("nrow и ncol должны быть > 0.")
        if not self.delr or len(self.delr) != int(self.ncol):
            raise ValueError("Длина delr должна равняться ncol.")
        if not self.delc or len(self.delc) != int(self.nrow):
            raise ValueError("Длина delc должна равняться nrow.")
        if self.xmin is not None and self.xmax is not None and self.xmin >= self.xmax:
            raise ValueError("Ожидается xmin < xmax.")
        if self.ymin is not None and self.ymax is not None and self.ymin >= self.ymax:
            raise ValueError("Ожидается ymin < ymax.")

    @staticmethod
    def _bounds_from_border(border: Polygon | MultiPolygon) -> tuple[Scalar, Scalar, Scalar, Scalar]:
        xmin, ymin, xmax, ymax = border.bounds
        return float(xmin), float(ymin), float(xmax), float(ymax)


class StructuredGridConfig(BaseGridConfig):
    type: Annotated[Literal[GridType.STRUCTURED], Field(GridType.STRUCTURED, exclude=True)]


class RefinementFeature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    geometry: object = Field(description="Путь к файлу или геометрия shapely")
    level: int = Field(ge=1, description="Уровень уточнения")

    @field_validator("geometry", mode="before")
    @classmethod
    def load_geometry(cls, v):
        """Поддержка: FilePath | shapely geometry | list[geometry]"""
        if v is None:
            raise ValueError("geometry не может быть None")

        if isinstance(v, shapely_base.BaseGeometry):
            return [v]
        if isinstance(v, list) and all(isinstance(g, shapely_base.BaseGeometry) for g in v):
            return v

        if isinstance(v, (str, Path)):
            path = Path(v)
            if not path.exists():
                raise FileNotFoundError(f"Файл геометрии не найден: {path}")

            gdf = gpd.read_file(path)
            if gdf.empty:
                raise ValueError(f"Векторный файл пуст: {path}")

            gdf = gdf.explode(index_parts=False)
            return list(gdf.geometry)

        raise TypeError(f"Некорректный тип geometry: {type(v)}")


class RefinementConfig(BaseModel):
    point: list[RefinementFeature] | None = None
    line: list[RefinementFeature] | None = None
    polygon: list[RefinementFeature] | None = None


class VertexGridConfig(BaseGridConfig):
    type: Annotated[Literal[GridType.VERTEX], Field(GridType.VERTEX, exclude=True)]
    gridgen_path: Annotated[
        DirectoryPath | FilePath | None, Field(default="grid", description="Путь до сохранения сетки gridgen")]
    gridgen_exe: Annotated[
        FilePath | None, Field(default="../../../bin/gridgen", description="Исполняемый файл Gridgen")]
    refinement: Annotated[RefinementConfig | None, Field(default=None, description="Параметры уточнения сетки")]


class UnstructuredGridConfig(BaseGridConfig):
    type: Annotated[Literal[GridType.UNSTRUCTURED], Field(GridType.UNSTRUCTURED, exclude=True)]


GridConfig = Annotated[
    StructuredGridConfig | VertexGridConfig | UnstructuredGridConfig,
    Field(discriminator='type')
]
