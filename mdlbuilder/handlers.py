import os
from typing import Union, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Polygon
from flopy.utils import Raster
from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.utils.geometry import Polygon


class GridHandler:
    def __init__(self, base):
        self.base = base
        self.model = base.model
        # if self.model.modelgrid is not None:
        if (self.base.is_mf6() and self.model.modelgrid._idomain is not None) or (self.base.is_mf2005() and self.model.modelgrid is not None):
            self.grid_poly = self._grid_polygons()

    @staticmethod
    def is_raster(val):
        return isinstance(val, str) and os.path.isfile(val) and val.endswith(".tif")

    @staticmethod
    def is_vector(val):
        return isinstance(val, str) and os.path.isfile(val) and val.endswith(".shp")

    @staticmethod
    def is_csv(val):
        return isinstance(val, str) and os.path.isfile(val) and val.endswith(".csv")

    @staticmethod
    def is_numeric(val):
        return isinstance(val, (int, float))

    def is_structured(self):
        return isinstance(self.model.modelgrid, StructuredGrid)

    def is_unstructured(self):
        return isinstance(self.model.modelgrid, UnstructuredGrid) or isinstance(self.model.modelgrid, VertexGrid)

    def remove_package(self, pckg_name):
        if self.model.has_package(pckg_name):
            self.model.remove_package(pckg_name)

    def const_grid(self, val):
        return val * np.ones(self.model.modelgrid.ncpl)

    def resample_raster(self, path: str, method="nearest") -> Any:
        rio = Raster.load(os.path.join(path))
        return rio.resample_to_grid(self.model.modelgrid, band=rio.bands[0], method=method)

    def _get_poly_arr(self):
        modelgrid = self.model.modelgrid
        poly_arr = modelgrid.map_polygons
        return poly_arr

    def _structured_grid_poly(self):
        polygons = []
        rowcol = []
        a, b = self._get_poly_arr()
        for i in range(len(a[0]) - 1):
            for j in range(len(a) - 1):
                poly = Polygon([
                    (a[j][i], b[j][i]),
                    (a[j][i + 1], b[j][i]),
                    (a[j][i + 1], b[j + 1][i]),
                    (a[j][i], b[j + 1][i]),
                    (a[j][i], b[j][i]),
                ])
                polygons.append(poly)
                rowcol.append((j, i))
        return rowcol, polygons

    def _unstructured_grid_poly(self):
        poly_arr = self._get_poly_arr()
        polygons = [Polygon(array.vertices) for array in poly_arr]
        return [], polygons

    def _grid_polygons(self) -> gpd.GeoDataFrame:
        if self.is_structured():
            rowcol, polygons = self._structured_grid_poly()
        else:
            rowcol, polygons = self._unstructured_grid_poly()
        griddf = gpd.GeoDataFrame(data=rowcol, columns=["row", "col"], geometry=polygons)
        griddf["geometry_y"] = griddf["geometry"]
        griddf = griddf.replace(np.nan, -999)
        return griddf

    def _get_row_grid(self, geometry):
        ncol = self.model.modelgrid.ncol
        ss_geo = zip([geometry['row']] * ncol, range(ncol))
        return pd.DataFrame(ss_geo, columns=["row", "col"])

    def _get_col_grid(self, geometry):
        nrow = self.model.modelgrid.nrow
        ss_geo = zip(range(nrow), [geometry['col']] * nrow)
        return pd.DataFrame(ss_geo, columns=["row", "col"])

    def _process_dict_geometry(self, geometry: dict) -> pd.DataFrame:
        if self.is_structured():
            if 'row' in geometry:
                return self._get_row_grid(geometry)
            elif 'col' in geometry:
                return self._get_col_grid(geometry)
        raise ValueError("Invalid geometry dictionary")

    def _process_intersections(self, layer):
        join_pdf = layer.sjoin(self.grid_poly, how="left").dropna(subset=["index_right"])
        return join_pdf.astype({"row": int, "col": int, "index_right": int})

    def _process_file_geometry(self, geometry: str) -> pd.DataFrame:
        layer = gpd.read_file(geometry)
        layer.columns = [x.lower() for x in layer.columns]
        return self._process_intersections(layer)

    def _process_file_csv(self, geometry: str) -> pd.DataFrame:
        layer = pd.read_csv(geometry)
        layer.columns = [x.lower() for x in layer.columns]
        layer = gpd.GeoDataFrame(layer, geometry=gpd.points_from_xy(layer.x, layer.y))
        return self._process_intersections(layer)

    def _get_all_structured_cells(self):
        return {"row": [i for i in range(self.model.modelgrid.ncol) for j in
                 range(self.model.modelgrid.nrow)],
                "col": [j for i in range(self.model.modelgrid.ncol) for j in
                 range(self.model.modelgrid.nrow)]}

    def _get_all_unstructured_cells(self):
        return {"index_right": [cell[0] for cell in self.model.modelgrid.cell2d]}

    def _process_geometry_all(self):
        if self.is_structured():
            results = pd.DataFrame(self._get_all_structured_cells())
        else:
            results = pd.DataFrame(self._get_all_unstructured_cells())
        return results

    def _process_geometry_shape(self, geometry):
        layer = gpd.GeoDataFrame({"geometry": geometry})
        return self._process_intersections(layer)

    def process_geometry(self, geometry: Union[dict, str, list]) -> pd.DataFrame:
        print(geometry)
        if isinstance(geometry, dict):
            return self._process_dict_geometry(geometry)
        elif self.is_vector(geometry):
            return self._process_file_geometry(geometry)
        elif self.is_csv(geometry):
            return self._process_file_csv(geometry)
        elif geometry == "all":
            return self._process_geometry_all()
        elif isinstance(geometry, list):
            return self._process_geometry_shape(geometry)
        else:
            raise ValueError("Unsupported geometry type")

    def is_in_idomain(self, row):
        if self.is_structured():
            if self.base.is_mf2005():
                if self.model.has_package("BAS6"):
                    domain = self.model.get_package("BAS6").ibound.array[0][(int(row.row), int(row.col))]
                else:
                    domain = "all"
                if domain != 0:
                    return True
            else:
                return self.model.modelgrid.idomain[0][(int(row.row), int(row.col))] == 1
        else:
            return self.model.modelgrid.idomain[0][int(row.index_right)] == 1
        return

