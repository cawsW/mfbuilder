from __future__ import annotations
from pathlib import Path

import numpy as np
from plpygis import Geometry
from flopy.utils import GridIntersect
from flopy.utils.gridgen import Gridgen
from flopy.discretization import StructuredGrid

from mfbuilder.dto.grid import RefinementFeature
from mfbuilder.mfmain import ProjectConfig
from mfbuilder.utils.mfdata import RasterHandler


class BaseGridBuilder:
    def __init__(self, ctx: ProjectConfig) -> None:
        self.ctx = ctx
        self.data = ctx.grid

    def _create_temp_dis(self):
        return StructuredGrid(
            delr=np.array(self.data.delr),
            delc=np.array(self.data.delc),
            top=np.ones((len(self.data.delc), len(self.data.delr))),
            botm=np.ones((self.data.nlay, len(self.data.delc), len(self.data.delr))) * (-10),
            xoff=self.data.xmin,
            yoff=self.data.ymin,
            nlay=self.data.nlay
        )

    def _read_surface(self, sfr, ncpl, modelgrid) -> np.ndarray:
        if isinstance(sfr, (int, float)):
            return np.ones(ncpl) * float(sfr)

        if isinstance(sfr, (str, Path)):
            path = sfr
            raster = RasterHandler(path)
            arr = raster.resample_to_grid(modelgrid)
            return arr
        raise TypeError(f"Некорректный тип поверхности: {type(sfr)}")

    def _process_surface(self, ncpl, modelgrid) -> (np.ndarray, np.ndarray):
        """Обрабатывает botm: вычисляет массивы всех поверхностей."""
        prev = self._read_surface(self.data.top, ncpl, modelgrid)
        layers = [prev.copy()]

        for layer in self.data.botm:
            key = "elev" if layer.elev is not None else \
                "thick_top" if layer.thick_top is not None else "thick_bot"
            val = getattr(layer, key)
            surface = self._read_surface(val, ncpl, modelgrid)

            if key == "elev":
                current = surface
            elif key == "thick_top":
                current = prev - surface
            elif key == "thick_bot":
                current = prev + surface
            else:
                raise ValueError(f"Недопустимый ключ: {key}")

            layers.append(current)
            prev = current

        srfs_rdc = RasterHandler.reduce_arrays(np.array(layers))
        if self.data.min_thickness > 0:
            srfs_rdc = RasterHandler.expand_arrays(srfs_rdc, self.data.min_thickness)

        return srfs_rdc[0], srfs_rdc[1:]


class StructuredGridBuilder(BaseGridBuilder):
    def _active_domain(self, grid: StructuredGrid):
        ix = GridIntersect(grid, method="vertex")
        result = ix.intersects(self.data.border)
        a_cells = np.zeros((self.data.nlay, self.data.nrow, self.data.ncol), dtype=np.int64)
        rows, cols = zip(*result.cellids)
        a_cells[:, rows, cols] = 1
        return a_cells


class VertexGridBuilder(BaseGridBuilder):
    def __init__(self, ctx: ProjectConfig):
        super().__init__(ctx)
        self.g = self._create_gridgen()

    def _create_gridgen(self):
        return Gridgen(self._create_temp_dis(), model_ws=self.data.gridgen_path, exe_name=self.data.gridgen_exe)

    def _active_domain(self) -> None:
        active_domain = Geometry(self.data.border.wkb).geojson["coordinates"]
        self.g.add_active_domain([active_domain], list(range(self.data.nlay)))

    def _add_refinement(self, features: list[RefinementFeature], geom_type: str) -> None:
        """Добавляет уточнение по уже валидированным геометриям."""
        for feature in features:
            print(feature.geometry)
            self.g.add_refinement_features(
                feature.geometry,
                geom_type,
                feature.level,
                list(range(self.data.nlay))
            )

    def _refinement_grid(self) -> None:
        """Обрабатывает все уровни уточнения из VertexGridConfig.refinement."""
        refinement = self.data.refinement
        if refinement.line:
            self._add_refinement(refinement.line, "line")
        if refinement.point:
            self._add_refinement(refinement.point, "point")
        if refinement.polygon:
            self._add_refinement(refinement.polygon, "polygon")

    def _get_gridprops(self):
        self._active_domain()
        if self.data.refinement:
            self._refinement_grid()
        self.g.build(verbose=False)
        return self.g.get_gridprops_disv()


class UnstructuredGridBuilder(BaseGridBuilder):
    pass
