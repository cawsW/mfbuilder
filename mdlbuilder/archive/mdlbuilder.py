import os
import platform
import stat
import math
from typing import Dict, List, Union, Any
from osgeo import gdal
from osgeo import ogr, osr

import numpy as np
import pandas as pd
import geopandas as gpd
from plpygis import Geometry
from shapely import Polygon
import flopy
import flopy.discretization as fgrid
from flopy.utils.gridgen import Gridgen
from flopy.utils import GridIntersect, Raster
from flopy.export.shapefile_utils import recarray2shp
from flopy.utils.mflistfile import Mf6ListBudget
from flopy.discretization import StructuredGrid

import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from flopy.utils.geometry import Polygon


class ConfigValidator:
    def __init__(self, config: dict, editing):
        self.config = config
        self.editing = editing

    def validate_config(self):
        self._validate_name()
        self._validate_workspace()
        self._validate_exe()
        self._validate_boundary()
        self._validate_grid_type()
        self._validate_cell_size()
        self._validate_nlay()
        self._validate_top()
        self._validate_botm()

    def _validate_name(self):
        if not self.config.get("name"):
            raise ValueError("No name specified")

    def _validate_workspace(self):
        workspace = self.config.get("workspace")
        if not workspace:
            raise ValueError("No workspace specified")
        if not os.path.exists(workspace):
            os.mkdir(workspace)

    def _validate_exe(self):
        exe = self.config.get("exe")
        if not exe:
            raise ValueError("No executable specified")
        if not os.path.exists(exe):
            raise ValueError(f"Executable {exe} does not exist")
        if platform.system() != "Windows":
            st = os.stat(exe)
            os.chmod(exe, st.st_mode | stat.S_IEXEC)

    def _validate_boundary(self):
        if not self.config.get("boundary"):
            raise ValueError("No boundary specified")

    def _validate_cell_size(self):
        if not self.config.get("cell_size"):
            raise ValueError("No cell size specified")

    def _validate_nlay(self):
        if not self.config.get("nlay"):
            raise ValueError("No number of layers specified")

    def _validate_top(self):
        if not self.config.get("top"):
            raise ValueError("No top specified")

    def _validate_botm(self):
        if not self.config.get("botm"):
            raise ValueError("No bottom specified")
        if len(self.botm) != self.nlay and not self.editing:
            raise ValueError("Number of botm layers does not match number of layers")

    def _validate_grid_type(self):
        if self.config.get("type"):
            typegrd = self.config.get("type")
            if typegrd != "structured":
                if self.config.get("gridgen_exe"):
                    gridgen_exe = self.config.get("gridgen_exe")
                    if platform.system() != "Windows":
                        st = os.stat(gridgen_exe)
                        os.chmod(gridgen_exe, st.st_mode | stat.S_IEXEC)
                else:
                    raise ValueError("No gridgen executable specified")
        else:
            raise ValueError("No grid type specified")


class ModelBase:
    def __init__(self, config: dict, editing: bool):
        self.config = config
        self.editing = editing
        self._initialize_config()
        self.simulation, self.model = self._initialize_model()

    def _initialize_config(self):
        validator = ConfigValidator(self.config)
        validator.validate_config()
        self.name = self.config.get("name")
        self.workspace = self.config.get("workspace")
        self.exe = self.config.get("exe")
        self.version = self.config.get("version", "mf6")
        self.steady = self.config.get("steady", True)
        self.perioddata = self.config.get("perioddata", [(1.0, 1, 1.0)])
        self.units = self.config.get("units", "DAYS")

    def _initialize_model(self):
        if self.editing:
            return self._load_existing_model()
        else:
            return self._create_new_model()

    def _create_new_model(self):
        if self.version != "mf6":
            sim = None
            gwf = flopy.modflow.Modflow(self.name, exe_name=self.exe, model_ws=self.workspace, version=self.version)
        else:
            sim = flopy.mf6.MFSimulation(
                sim_name=self.name, exe_name=self.exe, version="mf6", sim_ws=self.workspace
            )
            self._configure_simulation(sim)
            gwf = flopy.mf6.ModflowGwf(
                sim, modelname=self.name, model_nam_file=f"{self.name}.nam", save_flows=True, newtonoptions="NEWTON UNDER_RELAXATION"
            )
            self._configure_ims(sim)
        return sim, gwf

    def _load_existing_model(self):
        if self.version != "mf6":
            sim = None
            gwf = flopy.modflow.Modflow.load(f"{self.name}.nam", model_ws=self.workspace, version=self.version)
        else:
            sim = flopy.mf6.MFSimulation.load(sim_ws=self.workspace, exe_name=self.exe)
            if not self.steady and self.perioddata:
                sim.remove_package("tdis")
                self._configure_simulation(sim)
            self._configure_ims(sim)
            gwf = sim.get_model(self.name)
        return sim, gwf

    def _configure_simulation(self, sim):
        if self.steady:
            flopy.mf6.ModflowTdis(
                sim, pname="tdis", time_units=self.units, nper=1, perioddata=[(1.0, 1, 1.0)]
            )
        else:
            if not self.perioddata:
                raise ValueError("No time steps specified for transient model")
            sim.remove_package("tdis")
            flopy.mf6.ModflowTdis(
                sim, pname="tdis", time_units=self.units, nper=len(self.perioddata), perioddata=self.perioddata
            )

    def _configure_ims(self, sim):
        flopy.mf6.modflow.mfims.ModflowIms(
            sim, pname="ims", complexity="SIMPLE", inner_maximum=50, outer_maximum=25,
            backtracking_number=0, linear_acceleration="BICGSTAB", outer_dvclose=0.0001, inner_dvclose=0.00001
        )


class ModelAbstract:
    def __init__(self, base: ModelBase):
        self.base = base
        self.model = base.model

    def _grid_polygons(self):
        modelgrid = self.model.modelgrid
        poly_arr = modelgrid.map_polygons
        rowcol = []
        if type(modelgrid) is fgrid.StructuredGrid:
            polygons = []
            a, b = poly_arr
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
                    rowcol.append((i, j))
        else:
            polygons = [Polygon(array.vertices) for array in poly_arr]
        griddf = gpd.GeoDataFrame(data=rowcol, columns=["row", "col"], geometry=polygons)
        griddf = griddf.replace(np.nan, -999)
        # griddf = griddf.to_file('grid.shp')
        return griddf

    def raster_resample(self, path, method="nearest"):
        rio = Raster.load(os.path.join(path))
        data = rio.resample_to_grid(self.model.modelgrid, band=rio.bands[0], method=method)
        return data

    def _intersection_grid_attrs(self, geometry, attribute=[]):
        if type(geometry) is str:
            layer = gpd.read_file(os.path.join(geometry))
        elif type(geometry) is pd.DataFrame:
            geometry.columns = [col.lower() for col in geometry.columns]
            layer = gpd.GeoDataFrame(geometry, geometry=gpd.points_from_xy(geometry.x, geometry.y))
        else:
            layer = gpd.GeoDataFrame({"geometry": geometry})
        grid_poly = self._grid_polygons()
        join_pdf = layer.sjoin(grid_poly, how="left")
        join_pdf = join_pdf.dropna(subset=["index_right"])
        join_pdf = join_pdf.astype({"row": int, "col": int, "index_right": int})
        return join_pdf[["index_right", "row", "col"] + attribute]

    def _process_results(self, results, process_func, next=False):
        std = []
        results = results.drop_duplicates()
        for i, (idx, row) in enumerate(results.iterrows()):
            if type(self.model.modelgrid) is fgrid.StructuredGrid:
                el = process_func(row, "row", "col")
                if el:
                    std.append(el)
            else:
                if next:
                    if results.iloc[i].name != results.iloc[-1].name:
                        std.append(process_func([row, results.iloc[i + 1]], "index_right"))
                else:
                    std.append(process_func(row, "index_right"))
        return std


class ModelGrid(ModelAbstract):
    def __init__(self, model, config: dict, editing):
        super().__init__(model)
        self.proj_string = config.get("proj_string", "EPSG:3857")
        if config.get("type"):
            self.typegrd = config.get("type")
            if self.typegrd != "structured":
                self.refinement = config.get("refinement")
                if config.get("gridgen_exe"):
                    self.gridgen_exe = config.get("gridgen_exe")
                    # st = os.stat(self.gridgen_exe)
                    # os.chmod(self.gridgen_exe, st.st_mode | stat.S_IEXEC)
                else:
                    raise ValueError("No gridgen executable specified")
                self.gridgen_ws = config.get("gridgen_workspace", os.path.join("grid"))
        else:
            raise ValueError("No grid type specified")
        if config.get("boundary"):
            self.boundary = self.check_boundary(config.get("boundary"))
        else:
            raise ValueError("No boundary specified")
        if config.get("cell_size"):
            self.cell_size = config.get("cell_size")
        else:
            raise ValueError("No cell size specified")
        if config.get("nlay"):
            self.nlay = config.get("nlay")
        else:
            raise ValueError("No number of layers specified")
        if config.get("top"):
            self.top = config.get("top")
        else:
            raise ValueError("No top specified")
        if type(config.get("botm")) is list:
            self.botm = config.get("botm")
            if len(self.botm) != self.nlay and not editing:
                raise ValueError("Number of botm layers does not match number of layers")
        else:
            raise ValueError("No bottom specified")
        self.buffer = config.get("buffer", False)
        self.xmin, self.ymin, self.xmax, self.ymax = self.__init_bounds()
        self.min_thickness = config.get("min_thickness", 0.1)
        self.editing = editing
        # if not editing:
        self.create_dis()

    def check_boundary(self, boundary):
        if type(boundary) is str:
            boundary = gpd.read_file(os.path.join(boundary), crs=self.proj_string)
        elif type(boundary) is list:
            boundary = Polygon(boundary)
        return boundary

    def __init_bounds(self):
        if type(self.boundary) is gpd.GeoDataFrame:
            xmin, ymin, xmax, ymax = self.boundary.total_bounds
        else:
            xmin, ymin, xmax, ymax = self.boundary.bounds
        if self.buffer:
            buffer = self.cell_size * 2
            xmin, ymin, xmax, ymax = xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer
        return xmin, ymin, xmax, ymax

    def _init_size(self):
        n_row = math.ceil((self.xmax - self.xmin) / self.cell_size)
        n_col = math.ceil((self.ymax - self.ymin) / self.cell_size)
        delr = self.cell_size * np.ones(n_row, dtype=float)
        delc = self.cell_size * np.ones(n_col, dtype=float)
        return n_row, n_col, delr, delc

    def _clip_structure(self):
        n_row, n_col, delr, delc = self._init_size()
        sgr = fgrid.StructuredGrid(
            delc, delr, top=None, botm=None, xoff=self.xmin, yoff=self.ymin, angrot=0,
            # prj=self.proj_string,
            nlay=self.nlay
        )
        ix = GridIntersect(sgr, method="vertex")
        if type(self.boundary) is gpd.GeoDataFrame:
            result = ix.intersects(self.boundary.geometry[0])
        else:
            result = ix.intersects(self.boundary)
        idomain = np.zeros((self.nlay, n_col, n_row), dtype=np.int64)
        for lay in range(self.nlay):
            for cellid in result.cellids:
                idomain[lay][cellid] = 1
        return n_row, n_col, idomain

    def _prepare_top(self):
        if type(self.top) is not str:
            return self.top * np.ones(self.model.modelgrid.ncpl)
        else:
            return self.raster_resample(self.top)

    def _prepare_botm(self):
        for i, bot in enumerate(self.botm):
            if type(bot) is str:
                self.botm[i] = self.raster_resample(bot)
            elif type(bot) is dict:
                self.botm[i]["thick"] = bot["thick"] * np.ones(self.model.modelgrid.ncpl)
        if self.editing and len(self.botm) != self.nlay:
            for i in range(len(self.botm), self.nlay):
                self.botm.append(self.model.modelgrid.botm[i])
        return self.botm

    def _temp_simulation(self):
        n_row, n_col, delr, delc = self._init_size()
        ms = flopy.modflow.Modflow()
        dis = flopy.modflow.ModflowDis(
            ms,
            nlay=self.nlay,
            nrow=n_col,
            ncol=n_row,
            delr=delr,
            delc=delc,
            top=0,
            botm=-10,
            crs=self.proj_string,
            xul=self.xmin,
            yul=self.ymax
        )
        return ms

    def _add_refinement(self, geom, type_geom, usg):
        for data, level in geom:
            if type(data) is str:
                layer = gpd.read_file(os.path.join(data), crs=self.proj_string)
                layer = layer.explode()
                for row in layer.geometry.to_wkb():
                    obj = [Geometry(row).geojson["coordinates"]]
                    usg.add_refinement_features(obj, type_geom, level, list(range(self.nlay)))

    def _refinement_grid(self, usg):
        if self.refinement:
            points = self.refinement.get("point")
            lines = self.refinement.get("line")
            polygons = self.refinement.get("polygon")
            if lines:
                self._add_refinement(lines, "line", usg)
            if points:
                self._add_refinement(points, "point", usg)
            if polygons:
                self._add_refinement(polygons, "polygon", usg)

    def _gridgen(self):
        ms = self._temp_simulation()
        g = Gridgen(ms.modelgrid, model_ws=self.gridgen_ws, exe_name=self.gridgen_exe)
        active_domain = Geometry(self.boundary.geometry.to_wkb()[0]).geojson["coordinates"]
        g.add_active_domain([active_domain], list(range(self.nlay)))
        self._refinement_grid(g)
        g.build(verbose=False)
        return g.get_gridprops_disv()

    def create_dis(self):
        if not self.editing:
            if self.typegrd == "structured":
                n_row, n_col, idomain = self._clip_structure()
                if self.base.version == "mf6":
                    disv = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
                        self.model,
                        pname="dis",
                        nlay=self.nlay,
                        nrow=n_col,
                        ncol=n_row,
                        delr=self.cell_size,
                        delc=self.cell_size,
                        top=0,
                        botm=[-10 * i for i in range(self.nlay)],
                        idomain=idomain,
                        xorigin=self.xmin, yorigin=self.ymin, angrot=0
                    )
                else:
                    disv = flopy.modflow.ModflowDis(
                        self.model,
                        nlay=self.nlay,
                        nrow=n_col,
                        ncol=n_row,
                        delr=self.cell_size,
                        delc=self.cell_size,
                        top=0,
                        botm=[-10 * i for i in range(self.nlay)],
                        xul=self.xmin, yul=self.ymax,
                        crs=self.proj_string
                    )
                    bas = flopy.modflow.ModflowBas(self.model, ibound=np.array(idomain).reshape(disv.botm.shape),
                                                   strt=200)
            else:
                gridprops = self._gridgen()
                ncpl = gridprops["ncpl"]
                top = gridprops["top"]
                botm = gridprops["botm"]
                nvert = gridprops["nvert"]
                vertices = gridprops["vertices"]
                cell2d = gridprops["cell2d"]
                disv = flopy.mf6.modflow.mfgwfdisv.ModflowGwfdisv(
                    self.model,
                    nlay=self.nlay,
                    ncpl=ncpl,
                    top=top,
                    botm=botm,
                    nvert=nvert,
                    vertices=vertices,
                    cell2d=cell2d,
                    xorigin=self.xmin, yorigin=self.ymin, angrot=0
                )
        if not all(isinstance(x, float) for x in self.botm) or len(self.botm) == 0:
            if self.editing:
                disv = self.model.get_package("DISV")
            idomain, elevations, dem = self.form_idomain()
            if self.base.version == "mf6":
                disv.idomain = idomain
                disv.top = dem
                disv.botm = elevations
            else:
                disv.top = dem.reshape(disv.top.shape)
                disv.botm = np.array(elevations).reshape(disv.botm.shape)
        else:
            if self.editing:
                disv = self.model.get_package("DISV")
            disv.top = self.top
            disv.botm = self.botm

    def form_idomain(self):
        elevations, dem = self._reduce_geology()
        dom = np.array([dem, *elevations])
        idomain = []
        if self.model.modelgrid.idomain is not None:
            idomain_e = self.model.modelgrid.idomain
            for i, el in enumerate(dom[1:]):
                dom_nums = idomain_e[i].flatten()
                idomain.append(np.where((dom[i] - el == 0) & (dom_nums == 1), -1, dom_nums))
                # idomain[i] = np.where((dom[i] - el == 0) & (dom_nums == 1), -1, dom_nums).reshape(idomain[i].shape)
        else:
            for el in dom[1:]:
                idomain.append(np.where(dom[0] - el == 0, -1, 1))
        return idomain, elevations, dem

    def _reduce_geology(self):
        dem_data = self._prepare_top().flatten()
        bot_data = self._prepare_botm()
        for i, bot in enumerate(bot_data):
            if type(bot) is dict:
                if i == 0:
                    bot_data[i] = dem_data - bot["thick"]
                else:
                    bot_data[i] = bot_data[i - 1] - bot["thick"]
            else:
                bot_data[i] = bot.flatten()
                if i == 0:
                    bot_data[i] = np.where(dem_data - bot_data[i] < self.min_thickness, dem_data - self.min_thickness,
                                           bot_data[i])
                else:
                    bot_data[i] = np.where(bot_data[i - 1] - bot_data[i] < 0.1, bot_data[i - 1] - 0.1, bot_data[i])
        # bot_data = [dem_data - bot["thick"] if type(bot) is dict else bot for bot in bot_data]
        elevations = np.array([dem_data, *bot_data])[::-1]
        prepare_elev = []
        for lay in range(self.nlay):
            prepare_elev.append(np.minimum.reduce(elevations[lay:]))
        return prepare_elev[::-1], dem_data


class ModelParameters(ModelAbstract):
    def __init__(self, model, config: dict, editing):
        super().__init__(model)
        self.packages = config
        self.editing = editing
        self.add_packages()

    def _add_npf(self):
        npf = self.packages.get("npf")
        hk = npf.pop("k")
        for i, k in enumerate(hk):
            if type(k) is str:
                hk[i] = self.raster_resample(k)
        npf = flopy.mf6.ModflowGwfnpf(self.model, save_flows=True, k=hk, **npf)

    def _add_rch(self):
        rch = self.packages.get("rch")
        if self.base.version == "mf6":
            if type(rch) is list:
                rch_std = []
                for poly in rch:
                    result = self._intersection_grid_attrs(poly[0])
                    rch_std.append([[(0, node["row"], node["col"]), poly[1]] for idx, node in result.iterrows()])
                flopy.mf6.ModflowGwfrch(self.model, stress_period_data=[item for sublist in rch_std for item in sublist])
            elif type(rch) is float:
                flopy.mf6.ModflowGwfrcha(self.model, recharge={0: rch})
            elif type(rch) is str:
                rch = self.raster_resample(rch)
                flopy.mf6.ModflowGwfrcha(self.model, recharge={0: rch})
        else:
            flopy.modflow.ModflowRch(self.model, rech=rch)

    def _add_sto(self):
        options = self.packages.get("sto")
        steady_state = {step: True for step in options.get("steady_state")}
        transient = {step: True for step in options.get("transient")}
        flopy.mf6.ModflowGwfsto(self.model, steady_state=steady_state, transient=transient,
                                sy=options.get("sy"), ss=options.get("ss"), iconvert=options.get("iconvert"))

    def _add_ic(self):
        options = self.packages.get("ic")
        ic = flopy.mf6.ModflowGwfic(self.model, strt=options.get("head") if options else self.model.modelgrid.top)

    def _add_lpf(self):
        lpf = self.packages.get("lpf")
        hk = lpf.pop("k")
        flopy.modflow.ModflowLpf(self.model, hk=hk, **lpf)

    def add_packages(self):
        if not self.editing:
            if self.base.version == "mf6":
                self._add_ic()
        if self.packages.get("sto"):
            if self.editing:
                self.model.remove_package("STO")
            self._add_sto()
        if self.packages.get("npf"):
            if self.editing:
                self.model.remove_package("NPF")
            self._add_npf()
        if self.packages.get("rch"):
            if self.editing:
                self.model.remove_package("RCH")
            self._add_rch()
        if self.packages.get("lpf"):
            if self.editing:
                self.model.remove_package("LPF")
            self._add_lpf()


class ModelSourcesSinks(ModelAbstract):
    def __init__(self, model, config: dict, editing):
        super().__init__(model)
        self.packages = config
        self.editing = editing
        self.add_packages()

    def _ss_geometry(self, geometry):
        if type(self.model.modelgrid) is StructuredGrid:
            if type(geometry) is dict:
                if geometry.get("row") is not None:
                    ncol = self.model.modelgrid.ncol
                    ss_geo = zip([geometry.get("row")] * ncol, range(ncol))
                    results = pd.DataFrame(ss_geo, columns=["row", "col"])
                elif geometry.get("col") is not None:
                    nrow = self.model.modelgrid.nrow
                    ss_geo = zip(range(nrow), [geometry.get("col")] * nrow)
                    results = pd.DataFrame(ss_geo, columns=["row", "col"])
            elif type(geometry) is str:
                if os.path.isfile(geometry):
                    results = self._intersection_grid_attrs(geometry)
        return results

    def _add_source_sinks(self, pckg):

        def _ss_parameters(params):
            for name in params:
                std_atr = params[name]
                if type(std_atr) is str:
                    if os.path.isfile(std_atr):
                        par_val = self.raster_resample(std_atr)
                        if type(self.model.modelgrid) is StructuredGrid:
                            results[name] = par_val[results.row.values, results.col.values]
                        else:
                            results[name] = par_val[results.index_right]
                    elif std_atr == "top":
                        top = self.model.modelgrid.top
                        if type(self.model.modelgrid) is StructuredGrid:
                            results[name] = top[results.col.values, results.row.values]
                        else:
                            results[name] = top[results.index_right]
                elif type(std_atr) is list:
                    pass
                elif type(std_atr) is int or type(std_atr) is float:
                    results[name] = std_atr

        data = self.packages.get(pckg)
        step_std = {}
        for i, spd in data.items():
            all_results = []
            for option in spd:
                geometry = option.pop("geometry")
                layers = option.pop("layers")
                results = self._ss_geometry(geometry)
                if option.get("bname"):
                    bname = option.pop("bname")
                    results["bname"] = bname
                _ss_parameters(option)
                for li, lay in enumerate(layers):
                    if type(self.model.modelgrid) is StructuredGrid:
                        if self.base.version == "mf6":
                            all_results.extend([
                                [(lay - 1, row.row, row.col,), *row[list(option.keys())], row["bname"]]
                                for idx, row in results.iterrows()
                                if self.model.modelgrid.idomain[0][(int(row.row), int(row.col))] == 1
                            ])
                        else:
                            all_results.extend([
                                [lay - 1, row.col, row.row, *row[list(option.keys())]]
                                for idx, row in results.iterrows()
                                if self.model.get_package("BAS6").ibound.array[0][(int(row.col),int(row.row))] == 1
                            ])
                    else:
                        all_results.extend([
                            [(lay - 1, row.index_right), *row[list(option.keys())], row["bname"]]
                            for idx, row in results.iterrows()
                        ])
            all_results = self.delete_duplicates_grid_package(all_results)
            step_std[i] = all_results
        return step_std

            #     results = self._intersection_grid_attrs(option.get("geometry"))
            #     if type(option.get("head")) is str:
            #         heads = self.raster_resample(option.get("head"))
            #     else:
            #         heads = option.get("head")
            #     for lay in option["layers"]:
            #         all_results.extend(self._process_results(results, chd_func))
            # all_results = self.delete_duplicates_grid_package(all_results)
            # step_std[i] = all_results

    def _add_chd_test(self):
        return self._add_source_sinks("chd")

    def _add_riv_test(self):
        return self._add_source_sinks("riv")

    def _add_chd(self):
        data = self.packages.get("chd")

        def chd_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(int(row[args[1]]), int(row[args[0]]))] == 1:
                    if type(heads) is not int and type(heads) is not float:
                        head = heads[(row[args[1]], row[args[0]])]
                    else:
                        head = heads
                    return [(lay - 1, int(row[args[1]]), int(row[args[0]])), head]
            else:
                if type(heads) is not int and type(heads) is not float:
                    head = heads[row[args[0]]]
                else:
                    head = heads
                return [(lay - 1, int(row[args[0]])), head]

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                results = self._intersection_grid_attrs(option.get("geometry"))
                if type(option.get("head")) is str:
                    heads = self.raster_resample(option.get("head"))
                else:
                    heads = option.get("head")
                for lay in option["layers"]:
                    all_results.extend(self._process_results(results, chd_func))
            all_results = self.delete_duplicates_grid_package(all_results)
            step_std[i] = all_results

        return step_std

    def _add_river(self):
        data = self.packages.get("riv")

        def river_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(int(row[args[1]]), int(row[args[0]]))] == 1:
                    if stages == "top":
                        top = self.model.modelgrid.top[(int(row[args[1]]), int(row[args[0]]))]
                    else:
                        top = stages[(int(row[args[1]]), int(row[args[0]]))]
                    if self.base.version == "mf6":
                        return [(lay - 1, int(row[args[1]]), int(row[args[0]])), top, row["cond"], top - option["depth"],
                                f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']
                    else:
                        return [lay - 1, int(row[args[1]]), int(row[args[0]]), top, row["cond"],
                                top - option["depth"]]
            else:
                if stages == "top":
                    top = self.model.modelgrid.top[int(row[args[0]])]
                else:
                    top = stages[int(row[args[0]])]
                return [(lay - 1, int(row[args[0]])), top, row["cond"], top - option["depth"],
                        f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                attributes = []
                cond = None
                if type(option.get("cond")) is str:
                    cond = option.get("cond")
                    attributes.append(cond)
                results = self._intersection_grid_attrs(option.get("geometry"), attribute=attributes)
                if not cond:
                    results["cond"] = option.get("cond")
                if option.get("stage") != "top":
                    stages = self.raster_resample(option.get("stage"))
                else:
                    stages = option.get("stage")
                for lay in option["layers"]:
                    zz = 0
                    all_results.extend(self._process_results(results, river_func))
            last_occurrences = {}
            for item in all_results:
                if self.base.version =='mf6':
                    key = item[0]
                else:
                    key = (item[0], item[1], item[2])
                last_occurrences[key] = item
            unique_last_items = list(last_occurrences.values())
            step_std[i] = unique_last_items
        return step_std

    def _add_ghb(self):
        data = self.packages.get("ghb")

        def ghb_func(row, *args):
            if len(args) == 2:
                return [(lay - 1, int(row[args[1]]), int(row[args[0]])), row["head"], option["cond"][li],
                        f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']
            else:
                return [(lay - 1, int(row[args[0]])), row["head"], option["cond"][li],
                        f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                attributes = []
                cond = None
                if type(option.get("head")) is str:
                    cond = option.get("head")
                    attributes.append(cond)
                results = self._intersection_grid_attrs(option.get("geometry"), attributes)
                for li, lay in enumerate(option["layers"]):
                    if not cond:
                        heads = option.get("head")
                        if type(heads) is list:
                            results["head"] = heads[li]
                        else:
                            results["head"] = heads
                    all_results.extend(self._process_results(results, ghb_func))
            step_std[i] = all_results

        return step_std

    def _add_drn(self):
        data = self.packages.get("drn")

        def drn_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(int(row[args[1]]), int(row[args[0]]))] == 1:
                    head = self.model.modelgrid.top[(row[args[1]], row[args[0]])]
                    return [(lay - 1, int(row[args[1]]), int(row[args[0]])), head, option["cond"][li] if type(option["cond"]) is list else option["cond"],
                            f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']
            else:
                head = self.model.modelgrid.top[row[args[0]]]
                if type(option["stage"]) is not str:
                    head = option["stage"]
                else:
                    if option["stage"] != "top":
                        if "top" in option["stage"]:
                            head += float(option["stage"].replace("top", ""))
                        else:
                            head = stages[int(row[args[0]])]
                return [(lay - 1, row[args[0]]), head,  option["cond"][li] if type(option["cond"]) is list else option["cond"],
                        f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']

        step_std = {}
        if data.get("add"):
            std_ex = self.model.drn.stress_period_data.data[0].tolist()
            self.model.remove_package("DRN")

        for i, options in data.items():
            if type(i) is int:
                if options:
                    all_results = []
                    if data.get("add"):
                        all_results.extend(std_ex)
                    for option in options:
                        if not option.get("const"):
                            if type(option.get("stage")) is str and "top" not in option.get("stage"):
                                stages = self.raster_resample(option.get("stage"))
                            else:
                                stages = option.get("stage")
                            if option.get("geometry") == "all":
                                if type(self.model.modelgrid) is fgrid.StructuredGrid:
                                    results = pd.DataFrame({"row": [i for i in range(self.model.modelgrid.ncol) for j in
                                                                    range(self.model.modelgrid.nrow)],
                                                            "col": [j for i in range(self.model.modelgrid.ncol) for j in
                                                                    range(self.model.modelgrid.nrow)]})
                                else:
                                    results = pd.DataFrame({"index_right": [cell[0] for cell in self.model.modelgrid.cell2d], "name": 100})
                                    results.set_index("name", inplace=True)
                            else:
                                results = self._intersection_grid_attrs(option.get("geometry"))
                            for li, lay in enumerate(option["layers"]):
                                all_results.extend(self._process_results(results, drn_func))
                    last_occurrences = {}

                    for item in all_results:
                        key = item[0]
                        last_occurrences[key] = item
                    unique_last_items = list(last_occurrences.values())
                    step_std[i] = unique_last_items
                else:
                    step_std[i] = {}
        return step_std

    def pnt_along_line(self, line):
        # FIXME: change distance_delta considering modelgrid space
        distance_delta = 0.1
        distances = np.arange(0, line.length, distance_delta)
        points = [line.interpolate(distance) for distance in distances] + [line.boundary.geoms[1]]
        # multipoint = unary_union(points)
        return points

    def delete_duplicates_grid_package(self, example_list):
        unique_list = []
        seen_tuples = set()
        for sublist in example_list:
            if sublist[0] not in seen_tuples:
                unique_list.append(sublist)
                if self.base.version == "mf6":
                    seen_tuples.add(sublist[0])
                else:
                    seen_tuples.add((sublist[0], sublist[1], sublist[2]))
        return unique_list

    def _add_wel(self):
        data = self.packages.get("wel")

        def wel_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(int(row[args[1]]), int(row[args[0]]))] == 1:
                    return [(int(row["lay"]) - 1, int(row[args[1]]), int(row[args[0]])), -row["q"]]
            else:
                return [(int(row["lay"]) - 1, int(row[args[0]])), row["q"]]

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                attributes = []
                q, lay = None, None
                if type(option.get("q")) is str:
                    q = option.get("q")
                    attributes.append(q)
                if type(option.get("layers")) is str:
                    lay = option.get("layers")
                    attributes.append(lay)
                results = self._intersection_grid_attrs(option.get("geometry"), attribute=attributes)
                if not q:
                    results["q"] = option.get("q")
                else:
                    results.rename(columns={q: 'q'}, inplace=True)
                if not lay:
                    results["lay"] = option.get("layers")
                else:
                    results.rename(columns={lay: 'lay'}, inplace=True)
                results['index_count'] = results.groupby(results.index)['index_right'].transform('count')
                results["q"] = results["q"] / results["index_count"]
                results["q"] = results.groupby("index_right")["q"].transform('sum')
                all_results.extend(self._process_results(results, wel_func))
            step_std[i] = all_results
        return step_std

    def _add_hfb(self):
        data = self.packages.get("hfb")

        def hfb_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(int(row[args[1]]), int(row[args[0]]))] == 1:
                    # return [(lay - 1, int(row[args[1]]), int(row[args[0]])), -row["q"]]
                    pass
            else:
                return [[lay - 1, int(row[0][args[0]])], [lay - 1, int(row[1][args[0]])], hydchr]

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                hydchr = option["hydchr"]
                ln = gpd.read_file(os.path.join(option.get("geometry")))
                points = self.pnt_along_line(ln.iloc[0].geometry)
                results = self._intersection_grid_attrs(points)
                for lay in option["layers"]:
                    all_results.extend(self._process_results(results, hfb_func, next=True))
            step_std[i] = all_results
        return step_std

    def add_packages(self):
        riv = self.packages.get("riv")
        ghb = self.packages.get("ghb")
        drn = self.packages.get("drn")
        chd = self.packages.get("chd")
        wel = self.packages.get("wel")
        hfb = self.packages.get("hfb")

        if riv:
            if self.editing:
                self.model.remove_package("RIV")
            # step_std = self._add_riv_test()
            step_std = self._add_river()
            if self.base.version == "mf6":
                riv = flopy.mf6.ModflowGwfriv(self.model, stress_period_data=step_std, save_flows=True, boundnames=True)
            else:
                riv = flopy.modflow.ModflowRiv(self.model, stress_period_data=step_std)
        if ghb:
            if self.editing:
                self.model.remove_package("GHB")
            step_std = self._add_ghb()
            ghb = flopy.mf6.ModflowGwfghb(self.model, stress_period_data=step_std, boundnames=True)
        if drn:
            if self.editing:
                if not drn.get("add"):
                    self.model.remove_package("DRN")
            step_std = self._add_drn()
            drn = flopy.mf6.ModflowGwfdrn(self.model, stress_period_data=step_std, boundnames=True)
        if chd:
            if self.editing:
                self.model.remove_package("CHD")
            step_std = self._add_chd_test()
            chd = flopy.mf6.ModflowGwfchd(self.model, stress_period_data=step_std, boundnames=True)
        if wel:
            if self.editing:
                self.model.remove_package("WEL")
            step_std = self._add_wel()
            wel = flopy.mf6.ModflowGwfwel(self.model, stress_period_data=step_std)
        if hfb:
            if self.editing:
                self.model.remove_package("HFB")
            step_std = self._add_hfb()
            hfb = flopy.mf6.ModflowGwfhfb(self.model, stress_period_data=step_std)


class ModelObservations(ModelAbstract):
    def __init__(self, model, config: dict, editing):
        super().__init__(model)
        self.packages = config
        if editing:
            self.model.remove_package("OBS")
        self.add_packages()

    def _add_wells_obs(self):
        options = self.packages.get("wells")

        def wel_obs_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(row[args[1]], row[args[0]])] == 1:
                    return [(f"h{row['name']}", "HEAD", (row["lay"] - 1, row[args[1]], row[args[0]]))]
            else:
                return [(f"h{row['name']}", "HEAD", (row["lay"] - 1, row[args[0]]))]

        def get_lay_by_depth(row):
            if row["depth"]:
                for i, bot in enumerate(self.model.modelgrid.botm):
                    if row["depth"] <= self.model.modelgrid.top[int(row["index_right"])] - bot[int(row["index_right"])]:
                        return i + 1
            return self.model.modelgrid.nlay

        obslist = []
        obsdict = {}
        for option in options:
            attributes = []
            name, lay = None, None
            if option.get("data").endswith(".csv"):
                if type(option.get("name")) is str:
                    name = option.get("name")
                    attributes.append(name)
                if type(option.get("layers")) is str:
                    lay = option.get("layers")
                    attributes.append(lay)
                depth_at = option.get("depth")
                obs = pd.read_csv(os.path.join(option.get("data")))
                obs = obs.loc[:, obs.columns != 'lay']
                results = self._intersection_grid_attrs(obs, attribute=attributes)
                if not name:
                    results["name"] = option.get("name")
                else:
                    results.rename(columns={name: 'name'}, inplace=True)
                if not lay and not depth_at:
                    results["lay"] = option.get("layers")
                elif depth_at and not lay:
                    results["lay"] = results.merge(obs, on="name").apply(get_lay_by_depth, axis=1)
                    results_mod = results.merge(obs, on="name")
                    results_mod[list(obs.columns) + ["lay"]].to_csv(os.path.join(option.get("data")), index=False)
                else:
                    results.rename(columns={lay: 'lay'}, inplace=True)
                obslist = [obs_p[0] for obs_p in self._process_results(results, wel_obs_func)]

        obsdict[f"{self.model.name}.obs.head.csv"] = obslist
        obs = flopy.mf6.ModflowUtlobs(
            self.model, print_input=True, continuous=obsdict
        )

    def add_packages(self):
        if self.packages.get("wells"):
            self._add_wells_obs()


class ModelBuilder:
    def __init__(self, config: dict, external=False, editing=False):
        if config.get("base"):
            self.base = ModelBase(config.get("base"), editing)
        else:
            raise ValueError("No base options specified")

        if config.get("grid"):
            self.grid = ModelGrid(self.base, config.get("grid"), editing)
        else:
            if not editing:
                raise ValueError("No grid options specified")
        if config.get("parameters"):
            self.parameters = ModelParameters(self.base, config.get("parameters"), editing)
        else:
            print("No parameters options specified")
        self.sources = ModelSourcesSinks(self.base, config.get("sources"), editing) if config.get(
            "sources") else None
        self.observations = ModelObservations(self.base, config.get("observations"), editing) if config.get(
            "observations") else None
        self.external = external

    def output_package(self):
        model_name = self.base.name
        if self.base.version == "mf6":
            oc = flopy.mf6.ModflowGwfoc(
                self.base.model,
                pname="oc",
                budget_filerecord=f"{model_name}.cbb",
                head_filerecord=f"{model_name}.hds",
                headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
                saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
            )
        else:
            spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
            oc = flopy.modflow.ModflowOc(self.base.model, stress_period_data=spd, compact=True)
            pcg = flopy.modflow.ModflowPcg(self.base.model)

    def run(self):
        self.output_package()
        if self.base.version == "mf6":
            if self.external:
                self.base.simulation.set_all_data_external(True)
            self.base.simulation.write_simulation()
            success, buff = self.base.simulation.run_simulation()
        else:
            self.base.model.write_input()
            self.base.model.check()
            success, buff = self.base.model.run_model()

        if success:
            for line in buff:
                print(line)
        else:
            raise ValueError("Failed to run.")

    def create_raster(self, size, array, name, dir):
        crs = self.grid.proj_string
        range_x = (self.grid.xmax - self.grid.xmin) + 20 * size
        range_y = (self.grid.ymax - self.grid.ymin) + 20 * size
        num_cells_x = int(range_x / size)
        num_cells_y = int(range_y / size)
        if type(self.base.model.modelgrid) is fgrid.StructuredGrid:
            cell2d = self.base.model.modelgrid.xyzcellcenters
            centroids_x = cell2d[0].flatten()
            centroids_y = cell2d[1].flatten()
        else:
            cell2d = self.base.model.modelgrid.cell2d
            centroids_x = [cell[1] for cell in cell2d]
            centroids_y = [cell[2] for cell in cell2d]
        gridx, gridy = np.meshgrid(np.linspace(self.grid.xmin - 10 * size, self.grid.xmax + 10 * size, num=num_cells_x),
                                   np.linspace(self.grid.ymin - 10 * size, self.grid.ymax + 10 * size, num=num_cells_y))
        transform = from_origin(gridx.min(), gridy.max(), size, size)
        original_coords = np.array([centroids_x, centroids_y]).T
        interpolated_data = griddata(original_coords, array.flatten(), (gridx, gridy), method='linear')
        interpolated_data = np.where(interpolated_data > 1e6, np.nan, interpolated_data)
        raster_name = os.path.join(dir, f"{name}.tif")
        new_dataset = rasterio.open(
            raster_name,
            'w',
            driver='GTiff',
            height=interpolated_data.shape[0],
            width=interpolated_data.shape[1],
            count=1,
            dtype=str(interpolated_data.dtype),
            nodata=np.nan,
            crs=crs,
            transform=transform,
        )
        new_dataset.write(interpolated_data[::-1], 1)
        new_dataset.close()
        return raster_name

    def create_contours(self, name, dir, rst_name):
        indataset1 = gdal.Open(rst_name)
        sr = osr.SpatialReference(indataset1.GetProjection())
        in1 = indataset1.GetRasterBand(1)
        array = in1.ReadAsArray()
        demMax = np.nanmax(array)
        demMin = np.nanmin(array)
        contourPath = os.path.join(dir, f"{name}.shp")
        contourDs = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(contourPath)
        contourShp = contourDs.CreateLayer('contour', sr)

        # define fields of id and elev
        fieldDef = ogr.FieldDefn("ID", ogr.OFTInteger)
        contourShp.CreateField(fieldDef)
        fieldDef = ogr.FieldDefn("elev", ogr.OFTReal)
        contourShp.CreateField(fieldDef)
        conNum = 10
        conList = [round(x, 2) for x in np.linspace(demMin, demMax, conNum)]
        gdal.ContourGenerate(in1, 0, 0, conList, 0, 0.,
                             contourShp, 0, 1)
        contourDs.Destroy()

    def get_heads(self, per):
        headfile = f"{self.base.model.name}.hds"
        fname = os.path.join(self.base.model.model_ws, headfile)
        hds = flopy.utils.HeadFile(fname)
        head = hds.get_data(kstpkper=per)
        return head

    def get_npf(self):
        npf = self.base.model.get_package("NPF")
        return npf.k.array

    def get_rch(self):
        rch = self.base.model.get_package("RCH")
        return rch.recharge.array if rch else None

    @staticmethod
    def _create_dir(path):
        out_dir = os.path.join(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir

    def get_observations(self, directory):
        well_heads = self.observations.packages.get("wells")
        if well_heads:
            obs = pd.read_csv(os.path.join(well_heads[0].get("data")))
            obs["name"] = "H" + obs["name"].astype(str)
            model_obs = pd.read_csv(os.path.join(self.base.model.model_ws, f"{self.base.model.name}.obs.head.csv"))
            model_obs = model_obs.transpose().reset_index()
            print(model_obs)
            model_obs.columns = ["name", "head_model"]
            res = model_obs.merge(obs, on="name")
            res["residual"] = res["head"] - res["head_model"]
            res["rmse"] = ""
            res["rmse"][0] = (res.residual ** 2).mean() ** .5
            res.to_csv(os.path.join(directory, "residuals.csv"), index=False)

    def export_rasters(self):
        dir_rasters = self._create_dir("output/rasters")
        dir_contours = self._create_dir("output/rasters/contours")
        perioddata = self.base.simulation.get_package("tdis").perioddata.array
        ex_arr = []
        surf_arr = []
        ex_arr.extend(
            [(f"head_{i}{step}_lay_", self.get_heads((step, i))) for i, per in enumerate(perioddata) for step in
             range(per[1])])
        ex_arr.extend([("npf_lay_", self.get_npf())])
        rch = self.get_rch()
        if rch is not None:
            ex_arr.extend([("rch_lay_", rch)])
        surf_arr.extend([("bot_lay_", self.base.model.modelgrid.botm)])
        surf_arr.extend([("top_lay_", [self.base.model.modelgrid.top])])
        for name, arr in ex_arr:
            for lay, la in enumerate(arr):
                raster_name = self.create_raster(50, la, f"{name}{lay}", dir_rasters)
                self.create_contours(f"{name}{lay}", dir_contours, raster_name)

        for name, arr in surf_arr:
            for lay, la in enumerate(arr):
                raster_name = self.create_raster(50, la, f"{name}{lay}", dir_rasters)
                self.create_contours(f"{name}{lay}", dir_contours, raster_name)

    def export_vectors(self):
        dir_vectors = self._create_dir("output/vectors/")
        if self.base.steady:
            self.get_observations(dir_vectors)
        self.base.model.npf.export(os.path.join(dir_vectors, "npf.shp"))
        self.base.model.rch.export(os.path.join(dir_vectors, "rch.shp"))
        pcklist = [pn.upper().split('_')[0] for p in self.base.model.packagelist for pn in p.name]
        if "DRN" in pcklist:
            for per in range(len(self.base.perioddata)):
                drnstd = self.base.model.drn.stress_period_data.get_dataframe().get(per, None)
                if drnstd is not None:
                    self.dataframe_to_geom(drnstd, os.path.join(dir_vectors, f"drn{per}.shp"))
        if "RIV" in pcklist:
            for per in range(len(self.base.perioddata)):
                rivstd = self.base.model.riv.stress_period_data.get_dataframe().get(per)
                if rivstd is not None:
                    self.dataframe_to_geom(rivstd, os.path.join(dir_vectors, f"riv{per}.shp"))
        if "GHB" in pcklist:
            for per in range(len(self.base.perioddata)):
                ghbstd = self.base.model.ghb.stress_period_data.get_dataframe().get(per)
                if ghbstd is not None:
                    self.dataframe_to_geom(ghbstd, os.path.join(dir_vectors, f"ghb{per}.shp"))

    def export_other(self):
        dir_others = self._create_dir("output/others/")
        mf_list = Mf6ListBudget(os.path.join(self.base.model.model_ws, f"{self.base.model.name}.lst"))
        incrementaldf, cumulativedf = mf_list.get_dataframes()
        cumulativedf.to_csv(os.path.join(dir_others, "balances.csv"), index=False)

    def dataframe_to_geom(self, df, name):
        vertices = []
        for cell in df.cell:
            vertices.append(self.base.model.modelgrid.get_cell_vertices(cell))
        polygons = [Polygon(vrt) for vrt in vertices]
        recarray2shp(df.to_records(), geoms=polygons, shpname=name, crs=self.grid.proj_string)


class GeometryProcessor:
    def __init__(self, model):
        self.model = model

    def process_geometry(self, geometry: Union[dict, str]) -> pd.DataFrame:
        if isinstance(geometry, dict):
            return self._process_dict_geometry(geometry)
        elif isinstance(geometry, str) and os.path.isfile(geometry):
            return self._process_file_geometry(geometry)
        else:
            raise ValueError("Unsupported geometry type")

    def _process_dict_geometry(self, geometry: dict) -> pd.DataFrame:
        modelgrid = self.model.modelgrid
        if modelgrid.__class__.__name__ == 'StructuredGrid':
            if 'row' in geometry:
                ncol = modelgrid.ncol
                ss_geo = zip([geometry['row']] * ncol, range(ncol))
                return pd.DataFrame(ss_geo, columns=["row", "col"])
            elif 'col' in geometry:
                nrow = modelgrid.nrow
                ss_geo = zip(range(nrow), [geometry['col']] * nrow)
                return pd.DataFrame(ss_geo, columns=["row", "col"])
        raise ValueError("Invalid geometry dictionary")

    def _process_file_geometry(self, geometry: str) -> pd.DataFrame:
        layer = gpd.read_file(geometry)
        grid_poly = self._grid_polygons()
        join_pdf = layer.sjoin(grid_poly, how="left").dropna(subset=["index_right"])
        return join_pdf.astype({"row": int, "col": int, "index_right": int})

    def _grid_polygons(self) -> gpd.GeoDataFrame:
        modelgrid = self.model.modelgrid
        poly_arr = modelgrid.map_polygons
        rowcol = []

        if isinstance(modelgrid, StructuredGrid):
            polygons = []
            a, b = poly_arr
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
        else:
            polygons = [Polygon(array.vertices) for array in poly_arr]
        griddf = gpd.GeoDataFrame(data=rowcol, columns=["row", "col"], geometry=polygons)
        griddf = griddf.replace(np.nan, -999)
        return griddf

# Class to handle raster operations
class RasterProcessor:
    def __init__(self, model):
        self.model = model

    def resample(self, path: str, method="nearest") -> Any:
        rio = Raster.load(os.path.join(path))
        return rio.resample_to_grid(self.model.modelgrid, band=rio.bands[0], method=method)

# Class to handle source/sink parameters
class ParameterProcessor:
    def __init__(self, model):
        self.model = model
        self.raster_processor = RasterProcessor(model)
        self.geometry_processor = GeometryProcessor(model)

    def process_parameters(self, params: Dict[str, Any], results: pd.DataFrame) -> pd.DataFrame:
        for name, std_atr in params.items():
            if isinstance(std_atr, str) and os.path.isfile(std_atr):
                par_val = self.raster_processor.resample(std_atr)
                results[name] = par_val[results.row.values, results.col.values]
            elif std_atr == "top":
                top = self.model.modelgrid.top
                results[name] = top[results.row.values, results.col.values]
            elif isinstance(std_atr, (int, float)):
                results[name] = std_atr
        return results

# Main class to add source/sinks
class ModelSourcesSinksTest:
    def __init__(self, base: ModelBase, config: dict, editing):
        self.base = base
        self.model = base.model
        self.packages = config
        self.editing = editing
        self.geometry_processor = GeometryProcessor(base.model)
        self.parameter_processor = ParameterProcessor(base.model)
        self.add_packages()

    def add_packages(self):
        riv = self.packages.get("riv")

        if riv:
            if self.editing:
                self.model.remove_package("RIV")
            step_std = self._add_riv()
            if self.base.version == "mf6":
                riv = flopy.mf6.ModflowGwfriv(self.model, stress_period_data=step_std, save_flows=True, boundnames=True)
            else:
                riv = flopy.modflow.ModflowRiv(self.model, stress_period_data=step_std)

    def _add_source_sinks(self, package_name: str) -> Dict[int, List[Any]]:
        package_data = self.packages.get(package_name, {})
        step_std = {}

        for i, spd in package_data.items():
            all_results = []
            for option in spd:
                geometry = option.pop("geometry")
                layers = option.pop("layers")
                results = self.geometry_processor.process_geometry(geometry)

                if "bname" in option:
                    results["bname"] = option.pop("bname")

                results = self.parameter_processor.process_parameters(option, results)
                all_results.extend(self._prepare_results(results, layers, option))
            step_std[i] = self._delete_duplicates(all_results)
        return step_std

    def _prepare_results(self, results: pd.DataFrame, layers: List[int], option: Dict[str, Any]) -> List[Any]:
        all_results = []
        for li, lay in enumerate(layers):
            for idx, row in results.iterrows():
                if self.base.version == "mf6":
                    if self.model.modelgrid.idomain[0][(int(row.row), int(row.col))] == 1:
                        all_results.append([(lay - 1, row.row, row.col), *row[list(option.keys())]])
                else:
                    bas = self.model.get_package("BAS6")
                    if bas is not None:
                        domain = self.model.get_package("BAS6").ibound.array[0][(int(row.row), int(row.col))]
                    else:
                        domain = "all"
                    if domain == 1 or domain == "all":
                        all_results.append([lay - 1, row.row, row.col, *row[list(option.keys())]])
        return all_results

    def _delete_duplicates(self, results: List[Any]) -> List[Any]:
        unique_list = []
        seen_tuples = set()
        for sublist in results:
            if sublist[0] not in seen_tuples:
                unique_list.append(sublist)
                if self.base.version == "mf6":
                    seen_tuples.add(sublist[0])
                else:
                    seen_tuples.add((sublist[0], sublist[1], sublist[2]))
        return unique_list

    def _add_chd(self):
        return self._add_source_sinks("chd")

    def _add_riv(self):
        return self._add_source_sinks("riv")