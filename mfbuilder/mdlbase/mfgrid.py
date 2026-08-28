from __future__ import annotations
from pathlib import Path

import numpy as np
from plpygis import Geometry
from shapely.geometry import MultiPolygon
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

    def _idomain(self, top, botms) -> np.ndarray:
        """
        Генерирует массив idomain на основе толщины слоев.
        1 - если толщина слоя > 0
        -1 - если толщина слоя <= 0
        """
        upper_surfaces = np.concatenate([top[None, ...], botms[:-1]], axis=0)
        lower_surfaces = botms
        thickness = upper_surfaces - lower_surfaces
        idomain = np.where(thickness > 0, 1, -1).astype(np.int32)

        return idomain

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
        top, botms = srfs_rdc[0], srfs_rdc[1:]
        idomain = self._idomain(top, botms)
        return top, botms, idomain


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
        # Triangle/Voronoi не нуждается в объекте Gridgen — создаём его только
        # если реально будем строить сетку через gridgen (quadtree).
        self.g = self._create_gridgen() if self.data.method == "gridgen" else None

    def _create_gridgen(self):
        return Gridgen(self._create_temp_dis(), model_ws=self.data.gridgen_path, exe_name=self.data.gridgen_exe)

    def _active_domain(self) -> None:
        active_domain = Geometry(self.data.border.wkb).geojson["coordinates"]
        print(self.data.border.wkb)
        self.g.add_active_domain([active_domain], list(range(self.data.nlay)))

    def _add_refinement(self, features: list[RefinementFeature], geom_type: str) -> None:
        """Добавляет уточнение по уже валидированным геометриям."""
        for feature in features:
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

    def _get_gridprops_gridgen(self) -> dict:
        self._active_domain()
        if self.data.refinement:
            self._refinement_grid()
        self.g.build(verbose=False)
        return self.g.get_gridprops_disv()

    def _voronoi_refinement_regions(self):
        """Готовит (полигон, макс_площадь) для каждой зоны уточнения.

        point/line геометрия буферизуется в полигон (buffer или, по умолчанию,
        cell_size / 2**level). Площадь зоны = базовая площадь ячейки / 4**level —
        то же соотношение, что даёт один уровень quadtree-дробления в gridgen,
        чтобы одинаковый level давал сопоставимую густоту сетки в обоих методах.
        """
        base_area = float(self.data.cell_size) ** 2
        regions = []
        refinement = self.data.refinement
        if not refinement:
            return regions
        for features, is_area in ((refinement.polygon, True), (refinement.line, False), (refinement.point, False)):
            if not features:
                continue
            for feature in features:
                area = base_area / (4 ** feature.level)
                default_buffer = self.data.cell_size / (2 ** feature.level)
                for geom in feature.geometry:
                    poly = geom if is_area else geom.buffer(feature.buffer or default_buffer)
                    regions.append((poly, area))
        return regions

    def _get_gridprops_voronoi(self) -> dict:
        from flopy.utils.triangle import Triangle
        from flopy.utils.voronoi import VoronoiGrid

        tri = Triangle(model_ws=self.data.gridgen_path, exe_name=self.data.triangle_exe, angle=30)

        borders = list(self.data.border.geoms) if isinstance(self.data.border, MultiPolygon) else [self.data.border]
        for border_part in borders:
            tri.add_polygon(list(border_part.exterior.coords))

        regions = self._voronoi_refinement_regions()
        for poly, _ in regions:
            tri.add_polygon(list(poly.exterior.coords))

        base_area = float(self.data.cell_size) ** 2
        for i, border_part in enumerate(borders):
            pt = border_part.representative_point().coords[0]
            tri.add_region(pt, i, maximum_area=base_area)
        for i, (poly, area) in enumerate(regions, start=len(borders)):
            pt = poly.representative_point().coords[0]
            tri.add_region(pt, i, maximum_area=area)

        tri.build(verbose=False)
        vgrid = VoronoiGrid(tri)
        gridprops = vgrid.get_disv_gridprops()
        gridprops["nlay"] = self.data.nlay
        return gridprops

    def _get_gridprops(self) -> dict:
        if self.data.method == "voronoi":
            return self._get_gridprops_voronoi()
        return self._get_gridprops_gridgen()


class UnstructuredGridBuilder(BaseGridBuilder):
    pass
