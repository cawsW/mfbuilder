from __future__ import annotations
from pathlib import Path

from mfbuilder.mf6.mfgrid import VertexGridMf6Builder, StructuredGridMf6Builder
from mfbuilder.mf2005.mfgrid import StructuredGridMf2005Builder
from mfbuilder.mf6.mfpackages_flow import MF6FlowPackageBuilder
from mfbuilder.mf6.observations import MF6ObservationsBuilder
from mfbuilder.protocols import IModelBuilder, IGridBuilder, ISourceSinksBuilder
from mfbuilder.dto.types import EngineType, GridType, TGrid
from mfbuilder.dto.base import ProjectConfig
from mfbuilder.mf6.mfbuilder import MF6Builder
from mfbuilder.mf2005.mfbuilder import MF2005Builder
from mfbuilder.mf6.mfpackages import (MF6VertexRivHandler, MF6StructuredRivHandler,
                                      MF6VertexWelHandler, MF6VertexGhbHandler, MF6VertexDrnHandler)
from mfbuilder.mf2005.mfpackages import MF2005StructuredRivHandler
from mfbuilder.utils.mfdata import RasterHandler, VectorHandler


class BuilderFactory:
    def __init__(self) -> None:
        self._map = {
            EngineType.MF6: MF6Builder,
            EngineType.MF2005: MF2005Builder,
        }

    def create(self, ctx: ProjectConfig) -> IModelBuilder:
        cls = self._map.get(ctx.base.engine)
        if not cls:
            raise ValueError(f"Unsupported engine: {ctx.base.engine}")
        return cls(ctx)


class GridFactory:
    def __init__(self) -> None:
        self._map = {
            (EngineType.MF6, GridType.STRUCTURED): StructuredGridMf6Builder,
            (EngineType.MF6, GridType.VERTEX): VertexGridMf6Builder,
            (EngineType.MF2005, GridType.STRUCTURED): StructuredGridMf2005Builder,
        }

    def create(self, ctx: ProjectConfig) -> IGridBuilder:
        key = (ctx.base.engine, ctx.grid.type)
        cls = self._map.get(key)
        if not cls:
            raise ValueError(f"Unsupported grid: {ctx.grid.type}")
        return cls(ctx)


class FlowParametersFactory:
    def __init__(self) -> None:
        self._map = {
            EngineType.MF6: MF6FlowPackageBuilder,
            # EngineType.MF2005: MF2005Builder,
        }

    def create(self, model, grid, ctx: ProjectConfig):
        cls = self._map.get(ctx.base.engine)
        if not cls:
            raise ValueError(f"Unsupported engine: {ctx.base.engine}")
        return cls(model, grid, ctx)


class ObservationFactory:
    def __init__(self) -> None:
        self._map = {
            EngineType.MF6: MF6ObservationsBuilder,
            # EngineType.MF2005: MF2005Builder,
        }

    def create(self, model, grid, ctx: ProjectConfig):
        cls = self._map.get(ctx.base.engine)
        if not cls:
            raise ValueError(f"Unsupported engine: {ctx.base.engine}")
        return cls(model, grid, ctx)


class SourceSinksFactory:
    """Фабрика для создания обработчиков источников/стоков."""

    def __init__(self) -> None:
        self._map = {
        ("riv", EngineType.MF6, GridType.VERTEX): MF6VertexRivHandler,
        ("ghb", EngineType.MF6, GridType.VERTEX): MF6VertexGhbHandler,
        ("drn", EngineType.MF6, GridType.VERTEX): MF6VertexDrnHandler,
        ("wel", EngineType.MF6, GridType.VERTEX): MF6VertexWelHandler,
        ("riv", EngineType.MF6, GridType.STRUCTURED): MF6StructuredRivHandler,
        ("riv", EngineType.MF2005, GridType.STRUCTURED): MF2005StructuredRivHandler,
        }

    def create(self, ctx: ProjectConfig, name: str, grid) -> ISourceSinksBuilder[TGrid]:
        engine = ctx.base.engine
        grid_type = ctx.grid.type
        name = name.lower()
        key = (name, engine, grid_type)
        handler_cls = self._map.get(key)
        if not handler_cls:
            raise ValueError(f"Неизвестная комбинация: пакет={name}, движок={engine}, сетка={grid_type}")
        return handler_cls(grid, ctx.sources[name])


class DataFactory:
    """Фабрика для выбора обработчика по типу файла."""
    _raster_exts = {".tif", ".tiff", ".img", ".asc"}
    _vector_exts = {".shp", ".geojson", ".gpkg"}

    @classmethod
    def create(cls, path: str | Path):
        suffix = Path(path).suffix.lower()
        if suffix in cls._raster_exts:
            return RasterHandler(path)
        if suffix in cls._vector_exts:
            return VectorHandler(path)
        raise ValueError(f"Неизвестный тип файла: {path}")
