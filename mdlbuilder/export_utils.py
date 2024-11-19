import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import flopy
from flopy.export.shapefile_utils import recarray2shp
from flopy.discretization import StructuredGrid
from flopy.utils.mflistfile import Mf6ListBudget
from flopy.utils.geometry import Polygon
from scipy.interpolate import griddata
from shapely import Polygon
from osgeo import gdal, ogr, osr


class BaseExporter:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _create_dir(path: str) -> str:
        out_dir = os.path.join(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir

    def get_package_list(self):
        return [pn.upper().split('_')[0] for p in self.model.packagelist for pn in p.name]

    def get_heads(self, per):
        headfile = f"{self.model.name}.hds"
        fname = os.path.join(self.model.model_ws, headfile)
        hds = flopy.utils.HeadFile(fname)
        head = hds.get_data(kstpkper=per)
        return head


    def get_cbb(self, per):
        cbbfile = f"{self.model.name}.cbb"
        fname = os.path.join(self.model.model_ws, cbbfile)
        cbb = flopy.utils.CellBudgetFile(fname)
        riv = cbb.get_data(kstpkper=per, text="RIV")
        return riv

    def get_npf(self):
        npf = self.model.get_package("NPF")
        return npf.k.array

    def get_rch(self):
        rch = self.model.get_package("RCH")
        return rch.recharge.array if rch else None


class VectorExporter(BaseExporter):
    def __init__(self, model, crs, subspace="model"):
        super().__init__(model)
        self.crs = crs
        self.dir_vectors = self._create_dir(f"output/vectors/{subspace}")

    @staticmethod
    def _create_fields(contour_layer):
        fields = [("ID", ogr.OFTInteger), ("elev", ogr.OFTReal)]
        for name, field_type in fields:
            field_def = ogr.FieldDefn(name, field_type)
            contour_layer.CreateField(field_def)

    @staticmethod
    def _generate_contour_list(con_num, dem_min, dem_max):
        return [round(x, 2) for x in np.linspace(dem_min, dem_max, con_num)]

    @staticmethod
    def _generate_contours(contour_layer, con_list, band):
        gdal.ContourGenerate(band, 0, 0, con_list, 0, 0., contour_layer, 0, 1)

    def create_contours(self, name, path, rst_name, con_num=10):
        indataset = gdal.Open(rst_name)
        sr = osr.SpatialReference(indataset.GetProjection())
        band = indataset.GetRasterBand(1)
        data = band.ReadAsArray()
        dem_max = np.nanmax(data)
        dem_min = np.nanmin(data)
        contour_path = os.path.join(path, f"{name}.shp")
        contour_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(contour_path)
        contour_layer = contour_ds.CreateLayer('contour', sr)
        self._create_fields(contour_layer)
        con_list = self._generate_contour_list(con_num, dem_min, dem_max)
        self._generate_contours(contour_layer, con_list, band)
        contour_ds.Destroy()

    def _to_geom(self, df, name):
        vertices = []
        for cell in df.cellid_cell:
            vertices.append(self.model.modelgrid.get_cell_vertices(cell))
        polygons = [Polygon(vrt) for vrt in vertices]
        recarray2shp(df.to_records(), geoms=polygons, shpname=name, crs=self.crs)

    def _pkg_to_geom(self, pkg):
        for per in range(self.model.nper):
            std = self.model.get_package(pkg).stress_period_data.get_dataframe().get(per, None)
            if pkg == "RIV":
                dfcbb = pd.DataFrame(self.get_cbb((per, 0))[0])
                dfcbb["node"] = dfcbb["node"] - 1
                std = pd.merge(std, dfcbb, how="inner", left_on="cellid_cell", right_on="node")
            print(std)
            if std is not None:
                self._to_geom(std, os.path.join(self.dir_vectors, f"{pkg.lower()}{per}.shp"))

    def export_vectors(self):
        self.model.npf.export(os.path.join(self.dir_vectors, "npf.shp"))
        self.model.rch.export(os.path.join(self.dir_vectors, "rch.shp"))
        pkglist = self.get_package_list()
        exp_pkcg = ["DRN", "RIV", "GHB", "CHD"]
        for pkg in exp_pkcg:
            if pkg in pkglist:
                self._pkg_to_geom(pkg)


class RasterExporter(VectorExporter):
    def __init__(self, model, crs, extent, size=50, subspace="model"):
        super().__init__(model, crs)
        self.boundary = self.__init_boundary(extent)
        self.dir_rasters = self._create_dir(f"output/rasters/{subspace}")
        self.dir_contours = self._create_dir(f"output/rasters/{subspace}/contours")
        self.size = size
        self.xmin, self.ymin, self.xmax, self.ymax = self.__init_bounds()

    def __init_boundary(self, boundary):
        if type(boundary) is str and os.path.exists(boundary):
            boundary = gpd.read_file(os.path.join(boundary), crs=self.crs)
        elif type(boundary) is list:
            boundary = Polygon(boundary)
        return boundary

    def __init_bounds(self):
        if type(self.boundary) is gpd.GeoDataFrame:
            xmin, ymin, xmax, ymax = self.boundary.total_bounds
        else:
            xmin, ymin, xmax, ymax = self.boundary.bounds
        return xmin, ymin, xmax, ymax

    def get_num_cells(self):
        range_x = (self.xmax - self.xmin) + 20 * self.size
        range_y = (self.ymax - self.ymin) + 20 * self.size
        return int(range_x / self.size), int(range_y / self.size)

    def get_centroids(self):
        if isinstance(self.model.modelgrid, StructuredGrid):
            cell2d = self.model.modelgrid.xyzcellcenters
            centroids_x = cell2d[0].flatten()
            centroids_y = cell2d[1].flatten()
        else:
            cell2d = self.model.modelgrid.cell2d
            centroids_x = [cell[1] for cell in cell2d]
            centroids_y = [cell[2] for cell in cell2d]

        return np.array([centroids_x, centroids_y]).T

    def _create_grid(self):
        num_cells_x, num_cells_y = self.get_num_cells()
        gridx, gridy = np.meshgrid(
            np.linspace(self.xmin - 10 * self.size, self.xmax + 10 * self.size, num=num_cells_x),
            np.linspace(self.ymin - 10 * self.size, self.ymax + 10 * self.size, num=num_cells_y)
        )
        return gridx, gridy

    def _interpolate_data(self, array):
        gridx, gridy = self._create_grid()
        original_coords = self.get_centroids()
        transform = from_origin(gridx.min(), gridy.max(), self.size, self.size)
        interpolated_data = griddata(original_coords, array.flatten(), (gridx, gridy), method='linear')
        return np.where(interpolated_data > 1e6, np.nan, interpolated_data), transform

    def create_raster(self, interpolated_data, transform, raster_name):
        new_dataset = rasterio.open(
            raster_name,
            'w',
            driver='GTiff',
            height=interpolated_data.shape[0],
            width=interpolated_data.shape[1],
            count=1,
            dtype=str(interpolated_data.dtype),
            nodata=np.nan,
            crs=self.crs,
            transform=transform,
        )
        new_dataset.write(interpolated_data[::-1], 1)
        new_dataset.close()

    def _form_pkg_arrays(self):
        ex_arr = []
        ex_arr.extend(

            [(f"head_{i}_lay_", self.get_heads((self.model.modeltime.nstp[i] - 1, i))) for i, per in enumerate(range(self.model.nper))]
            # [(f"head_{i}{step}_lay_", self.get_heads((step, i))) for i, per in enumerate(range(self.model.nper)) for step in
            #  range(self.model.modeltime.nstp[i])]
        )

        ex_arr.extend([("npf_lay_", self.get_npf())])
        rch = self.get_rch()
        if rch is not None:
            ex_arr.extend([("rch_lay_", rch)])
        return ex_arr

    def _form_surface_arrays(self):
        surf_arr = []
        surf_arr.extend([("bot_lay_", self.model.modelgrid.botm)])
        surf_arr.extend([("top_lay_", [self.model.modelgrid.top])])
        return surf_arr

    def arrays_to_raster(self, arrays):
        for name, arr in arrays:
            for lay, la in enumerate(arr):
                raster_name = os.path.join(self.dir_rasters, f"{name}{lay}.tif")
                interpolate_data, transform = self._interpolate_data(la)
                self.create_raster(interpolate_data, transform, raster_name)
                self.create_contours(f"{name}{lay}", self.dir_contours, raster_name)

    def export_rasters(self):
        pkg_arr = self._form_pkg_arrays()
        surf_arr = self._form_surface_arrays()
        self.arrays_to_raster(pkg_arr)
        self.arrays_to_raster(surf_arr)


class TxtExporter(BaseExporter):
    def __init__(self, model, subspace="model"):
        super().__init__(model)
        self.dir_txt = self._create_dir(f"output/txt/{subspace}")

    @staticmethod
    def get_real_obs(path):
        dirname = os.path.dirname(path)
        filename = f"{Path(path).stem}_text.csv"
        obs = pd.read_csv(os.path.join(dirname, filename))
        return obs

    @staticmethod
    def calc_obs_err(res: pd.DataFrame):
        res["residual"] = res["head"] - res["head_model"]
        res["rmse"] = ""
        res["rmse"][0] = (res.residual ** 2).mean() ** .5
        return res

    def get_model_obs(self):
        model_obs = pd.read_csv(
            os.path.join(self.model.model_ws, f"{self.model.name}.obs.head.csv"), encoding="latin")
        model_obs = model_obs.transpose().reset_index()
        model_obs.columns = ["name", "head_model"]
        return model_obs

    def export_observations(self, observations):
        real_data = observations.get("wells")
        if real_data:
            for i, data in enumerate(real_data):
                obs = self.get_real_obs(data.get("geometry"))
                model_obs = self.get_model_obs()
                res = model_obs.merge(obs, on="name")
                res = self.calc_obs_err(res)
                res.to_csv(os.path.join(self.dir_txt, f"residuals_{i + 1}.csv"), index=False)

    def export_balance(self):
        mf_list = Mf6ListBudget(os.path.join(self.model.model_ws, f"{self.model.name}.lst"))
        incrementaldf, cumulativedf = mf_list.get_dataframes()
        cumulativedf.to_csv(os.path.join(self.dir_txt, "balances_cumm.csv"))
        incrementaldf.to_csv(os.path.join(self.dir_txt, "balances_inc.csv"))

    def export_all(self, observations):
        self.export_balance()
        self.export_observations(observations)
