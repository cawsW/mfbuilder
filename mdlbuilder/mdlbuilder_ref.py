import os
import re
import math
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import flopy
import flopy.discretization as fgrid
from flopy.utils.gridgen import Gridgen
from flopy.utils import GridIntersect
from plpygis import Geometry
from shapely import Polygon
from statsmodels.graphics.tukeyplot import results

from mdlbuilder.validators import ConfigValidator
from mdlbuilder.handlers import GridHandler


class ModelBase:
    def __init__(self, config: dict, editing: bool):
        self.name = config.get("name")
        self.workspace = config.get("workspace")
        self.exe = config.get("exe")
        self.version = config.get("version", "mf6")
        self.steady = config.get("steady", True)
        self.perioddata = config.get("perioddata", [(1.0, 1, 1.0)])
        self.units = config.get("units", "DAYS")
        self.editing = editing
        self.simulation, self.model = self._initialize_model()

    def is_mf6(self):
        return self.version == "mf6"

    def is_mf2005(self):
        return self.version == "mf2005"

    def _initialize_model(self):
        if self.editing:
            return self._load_existing_model()
        else:
            return self._create_new_model()

    def _create_new_model(self):
        if self.is_mf6():
            sim = flopy.mf6.MFSimulation(
                sim_name=self.name, exe_name=self.exe, version="mf6", sim_ws=self.workspace
            )
            self._configure_simulation(sim)
            gwf = flopy.mf6.ModflowGwf(
                sim, modelname=self.name, model_nam_file=f"{self.name}.nam", save_flows=True,
                newtonoptions="NEWTON UNDER_RELAXATION"
            )
            self._configure_ims(sim)
        else:
            sim = None
            gwf = flopy.modflow.Modflow(self.name, exe_name=self.exe, model_ws=self.workspace, version=self.version)
        return sim, gwf

    def _load_existing_model(self):
        if self.is_mf6():
            sim = flopy.mf6.MFSimulation.load(sim_ws=self.workspace, exe_name=self.exe)
            self._configure_simulation(sim)
            self._configure_ims(sim)
            gwf = sim.get_model(self.name)
        else:
            sim = None
            gwf = flopy.modflow.Modflow.load(f"{self.name}.nam", model_ws=self.workspace, version=self.version)
        return sim, gwf

    def _configure_simulation(self, sim):
        if not self.steady or self.editing:
            sim.remove_package("tdis")
        flopy.mf6.ModflowTdis(
            sim, pname="tdis", time_units=self.units, nper=len(self.perioddata), perioddata=self.perioddata
        )

    def _configure_ims(self, sim):
        flopy.mf6.modflow.mfims.ModflowIms(
            sim, pname="ims", complexity="SIMPLE", inner_maximum=500, outer_maximum=250,
            backtracking_number=0, linear_acceleration="BICGSTAB", outer_dvclose=0.0001, inner_dvclose=0.00001,

        )


class MfBaseGrid(GridHandler):
    def __init__(self, base: ModelBase, config: dict, editing):
        super().__init__(base)
        self.editing = editing
        self.proj_string = config.get("proj_string", "EPSG:3857")
        self.boundary = self.__init_boundary(config.get("boundary"))
        self.cell_size = config.get("cell_size")
        self.nlay = config.get("nlay")
        self.min_thickness = config.get("min_thickness", 0)
        self.top = config.get("top")
        self.botm = config.get("botm")
        self.xmin, self.ymin, self.xmax, self.ymax = self.__init_bounds()
        self.n_row, self.n_col, self.delr, self.delc = self.__init_size()

    def __init_boundary(self, boundary):
        if type(boundary) is str and os.path.exists(boundary):
            boundary = gpd.read_file(os.path.join(boundary), crs=self.proj_string)
        elif type(boundary) is list:
            boundary = Polygon(boundary)
        return boundary

    def __init_bounds(self):
        if type(self.boundary) is gpd.GeoDataFrame:
            xmin, ymin, xmax, ymax = self.boundary.geometry[0].bounds
        else:
            xmin, ymin, xmax, ymax = self.boundary.bounds
        return xmin, ymin, xmax, ymax

    def __init_size(self):
        n_row = math.floor((self.ymax - self.ymin) / self.cell_size)
        n_col = math.floor((self.xmax - self.xmin) / self.cell_size)
        delr = self.cell_size * np.ones(n_row, dtype=float)
        delc = self.cell_size * np.ones(n_col, dtype=float)
        return n_row, n_col, delr, delc

    @staticmethod
    def is_dict_with_thick(bot):
        return isinstance(bot, dict) and "thick" in bot

    @staticmethod
    def is_const_botm(bot):
        return bot == "const"

    def _prepare_top(self, top):
        if self.is_raster(top):
            return self.resample_raster(top)
        else:
            return self.const_grid(top)

    def __handle_raster_botm(self, botm, lay, bot):
        botm[lay] = self.resample_raster(bot)
        if lay == 0:
            botm[lay] = np.where(self.top - botm[lay] <= self.min_thickness, self.top - self.min_thickness, botm[lay])
        else:
            botm[lay] = np.where(botm[lay - 1] - botm[lay] <= self.min_thickness, botm[lay - 1] - self.min_thickness,
                                 botm[lay])
        return botm[lay]

    def __handle_thick_dict(self, botm, lay, bot):
        botm[lay]["thick"] = self.const_grid(bot["thick"])
        if lay == 0:
            return self.top - botm[lay]["thick"]
        else:
            return botm[lay - 1] - botm[lay]["thick"]

    def _prepare_botm(self, botm):
        for lay, bot in enumerate(botm):
            if self.is_raster(bot):
                botm[lay] = self.__handle_raster_botm(botm, lay, bot)
            elif self.is_dict_with_thick(bot):
                botm[lay] = self.__handle_thick_dict(botm, lay, bot)
            elif self.is_numeric(bot):
                botm[lay] = self.const_grid(bot)
            elif self.is_const_botm(bot):
                botm[lay] = self.model.modelgrid.botm[lay]
            else:
                raise ValueError("Invalid type of botm")
        return botm

    def _reduce_geology(self):
        self.top = self._prepare_top(self.top)
        self.botm = self._prepare_botm(self.botm)
        elevations = np.array([self.top, *self.botm])[::-1]
        prepare_elev = [np.minimum.reduce(elevations[lay:]) for lay in range(self.nlay)]
        return prepare_elev[::-1]

    def _change_rdc_domain(self, surfaces, cells):
        idomain_rdc = []
        for i, el in enumerate(surfaces[1:]):
            dom_nums = cells[i]
            idomain_rdc.append(
                np.where((surfaces[i] - el == 0) & (dom_nums == 1), -1 if self.base.is_mf6() else 0, dom_nums)
            )
        return idomain_rdc

    def _create_rdc_domain(self, surfaces):
        idomain_rdc = []
        for i, el in enumerate(surfaces[1:]):
            idomain_rdc.append(np.where(surfaces[i] - el == 0, -1 if self.base.is_mf6() else 0, 1))
        return idomain_rdc

    def _form_idomain(self, cells=None):
        rdc_elv = self._reduce_geology()
        srf = np.array([self.top, *rdc_elv])
        if cells is not None:
            idomain = self._change_rdc_domain(srf, cells)
        else:
            idomain = self._create_rdc_domain(srf)
        return idomain, rdc_elv


class MfStructuredGrid(MfBaseGrid):
    def __init__(self, base, config: dict, editing):
        super().__init__(base, config, editing)
        self.create_dis()

    def _active_cells(self, ix_cells):
        a_cells = np.zeros((self.nlay, self.n_row, self.n_col), dtype=np.int64)
        rows, cols = zip(*ix_cells.cellids)
        a_cells[:, rows, cols] = 1
        return a_cells

    def _get_intersecting_cells(self):
        sgr = fgrid.StructuredGrid(
            self.delr, self.delc, top=None, botm=None, xoff=self.xmin, yoff=self.ymin, angrot=0, nlay=self.nlay,
        )
        return GridIntersect(sgr, method="vertex")

    def _clip_structure(self):
        ix = self._get_intersecting_cells()
        if type(self.boundary) is gpd.GeoDataFrame:
            result = ix.intersects(self.boundary.geometry[0])
        else:
            result = ix.intersects(self.boundary)
        return self._active_cells(result)

    def _create_dis_mf2005(self):
        print(self.xmin, self.ymin)
        return flopy.modflow.ModflowDis(
            self.model,
            nlay=self.nlay,
            nrow=self.n_row,
            ncol=self.n_col,
            delr=self.cell_size,
            delc=self.cell_size,
            perlen=[per[0] for per in self.base.perioddata],
            nstp=[stp[1] for stp in self.base.perioddata],
            tsmult=[ts[2] for ts in self.base.perioddata],
            steady=True if self.base.steady else [True] + self.base.steady * (len(self.base.perioddata) - 1),
            top=10,
            botm=[lay * -10 for lay in range(self.nlay)],
            xul=self.xmin, yul=self.ymax,
            crs=self.proj_string
        )

    def _create_bas(self, idomain):
        if self.editing:
            self.model.remove_package("BAS6")
        bas = flopy.modflow.ModflowBas(self.model, ibound=np.array(idomain), strt=400)
        return bas

    def _create_dis_mf6(self, a_cells):
        return flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
            self.model,
            pname="dis",
            nlay=self.nlay,
            nrow=self.n_row,
            ncol=self.n_col,
            delr=self.cell_size,
            delc=self.cell_size,
            top=0,
            idomain=a_cells,
            botm=[lay * -10 for lay in range(self.nlay)],
            xorigin=self.xmin, yorigin=self.ymin, angrot=0,
        )

    def _edit_dis_mf2005(self, dis, botm):
        dis.top = self.top
        dis.botm = botm
        return dis

    def _edit_dis_mf6(self, dis, botm, idomain):
        dis.top = self.top
        dis.botm = botm
        dis.idomain = idomain
        return dis

    def create_dis(self):
        a_cells = self._clip_structure()
        if not self.editing:
            if self.base.is_mf6():
                dis = self._create_dis_mf6(a_cells)
            else:
                dis = self._create_dis_mf2005()
        else:
            dis = self.model.get_package("DIS")
        idomain, elevations = self._form_idomain(a_cells)
        if self.base.is_mf6():
            dis = self._edit_dis_mf6(dis, elevations, idomain)
        if self.base.is_mf2005():
            dis = self._edit_dis_mf2005(dis, elevations)
            bas = self._create_bas(a_cells)


class MfUnstructuredGrid(MfBaseGrid):
    def __init__(self, base, config: dict, editing):
        super().__init__(base, config, editing)
        self.refinement = config.get("refinement")
        self.gridgen_exe = config.get("gridgen_exe")
        self.gridgen_ws = config.get("gridgen_workspace", os.path.join("grid"))
        self.create_dis()

    def _add_refinement(self, geom, type_geom, usg):
        for data in geom:
            geometry = data["geometry"]
            level = data["level"]
            if self.is_vector(geometry):
                layer = gpd.read_file(os.path.join(geometry), crs=self.proj_string)
                layer = layer.explode()
                usg.add_refinement_features(list(layer.geometry), type_geom, level, list(range(self.nlay)))

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

    def _active_domain(self, g):
        active_domain = Geometry(self.boundary.geometry.to_wkb()[0]).geojson["coordinates"]
        g.add_active_domain([active_domain], list(range(self.nlay)))

    def _gridgen(self):
        ms = self._temp_dis()
        g = Gridgen(ms.modelgrid, model_ws=self.gridgen_ws, exe_name=self.gridgen_exe)
        self._active_domain(g)
        self._refinement_grid(g)
        g.build(verbose=False)
        return g.get_gridprops_disv()

    def _temp_dis(self):
        ms = flopy.modflow.Modflow()
        dis = flopy.modflow.ModflowDis(
            ms,
            nlay=self.nlay,
            nrow=self.n_row,
            ncol=self.n_col,
            delr=self.delc,
            delc=self.delr,
            top=0,
            botm=-10,
            xul=self.xmin,
            yul=self.ymax,
        )
        return ms

    def _create_disv(self):
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
        )
        poly_arr = self.model.modelgrid.map_polygons
        return disv

    def create_dis(self):
        if not self.editing:
            disv = self._create_disv()
            idomain, elevations = self._form_idomain()
            disv.top = self.top
            disv.botm = elevations
            disv.idomain = idomain
        else:
            disv = self.model.get_package("DISV")


class ModelParameters(GridHandler):
    def __init__(self, base: ModelBase, config: dict, editing):
        super().__init__(base)
        self.packages = config
        self.editing = editing
        if self.base.is_mf6():
            self.add_packages_mf6()
        if self.base.is_mf2005():
            self.add_packages_mf2005()

    def _package_geometry(self, pars, default, data):
        results = self.process_geometry(pars)
        if self.base.is_mf6():
            par_array = np.ones(self.base.model.modelgrid.ncpl) * default
            results = results.groupby('index_right')['dw'].mean().reset_index()
            for i, row in results.iterrows():
                if data:
                    par_array[int(row.index_right)] = row[data]
                else:
                    raise ValueError("set column in geometry file")
        else:
            par_array = np.ones(self.base.model.modelgrid.idomain[0].shape) * default
            par_array = par_array
            results = results.groupby(['row', 'col'])['dw'].mean().reset_index()
            for i, row in results.iterrows():
                if data:
                    par_array[(int(row.row), int(row.col))] = row[data]
                else:
                    raise ValueError("set column in geometry file")
        return par_array

    def _add_package(self, pkg):
        default = pkg.pop("default") if pkg.get("default") else 0
        data = pkg.pop("data") if pkg.get("data") else None
        for pars_name in pkg:
            pars = pkg[pars_name]
            if isinstance(pars, list):
                for i, par in enumerate(pars):
                    if self.is_raster(par):
                        pkg[pars_name][i] = self.resample_raster(par)
            elif self.is_numeric(pars):
                pkg[pars_name] = pars
            elif self.is_vector(pars):
                pkg[pars_name] = self._package_geometry(pars, default, data)
        return pkg

    def add_packages_mf2005(self):
        if self.packages.get("lpf"):
            if self.editing:
                self.model.remove_package("LPF")
            lpf_data = self._add_package(self.packages.get("lpf"))
            lpf = flopy.modflow.ModflowLpf(self.model, **lpf_data)
        if self.packages.get("rch"):
            if self.editing:
                self.model.remove_package("RCH")
            rch_data = self._add_package(self.packages.get("rch"))
            if type(rch_data.get("rech")) is np.ndarray:
                rch_data["rech"] = rch_data["rech"].reshape((self.model.modelgrid.nrow, self.model.modelgrid.ncol))
            rch = flopy.modflow.ModflowRch(self.model, **rch_data)
        if self.packages.get("bcf"):
            if self.editing:
                self.model.remove_package("BCF")
            bcf_data = self._add_package(self.packages.get("bcf"))
            bcf = flopy.modflow.ModflowBcf(self.model, **bcf_data)

    def add_packages_mf6(self):
        if not self.editing:
            ic_data = self._add_package(self.packages.get("ic"))
            ic = flopy.mf6.ModflowGwfic(self.model, **ic_data)
        if self.packages.get("npf"):
            if not self.editing:
                npf_data = self._add_package(self.packages.get("npf"))
                npf = flopy.mf6.ModflowGwfnpf(self.model, save_flows=True, **npf_data)
        if self.packages.get("rch"):
            if not self.editing:
                rch_data = self._add_package(self.packages.get("rch"))
                rch = flopy.mf6.ModflowGwfrcha(self.model, recharge={0: rch_data["rech"]})
        if self.packages.get("sto"):
            if self.editing:
                self.model.remove_package("STO")
            self._add_sto()

    def _add_sto(self):
        options = self.packages.get("sto")
        options["steady_state"] = {step: True for step in options.get("steady_state")}
        options["transient"] = {step: True for step in options.get("transient")}
        flopy.mf6.ModflowGwfsto(self.model, **options)


class ModelSourcesSinks(GridHandler):
    def __init__(self, base: ModelBase, config: dict, editing):
        super().__init__(base)
        self.packages = config
        self.editing = editing
        self.add_packages()

    @staticmethod
    def separate_string_and_number(s):
        match = re.match(r"([a-zA-Z\s]+)([+-]\s*\d+)", s)
        if match:
            string_part = match.group(1).strip()
            number_part = match.group(2).replace(" ", "")
            return string_part, number_part
        else:
            return None, None

    def add_packages(self):
        riv = self.packages.get("riv")
        chd = self.packages.get("chd")
        ghb = self.packages.get("ghb")
        drn = self.packages.get("drn")
        wel = self.packages.get("wel")
        hfb = self.packages.get("hfb")
        if riv:
            self._add_riv()
        if chd:
            self._add_chd()
        if ghb:
            self._add_ghb()
        if drn:
            self._add_drn()
        if wel:
            self._add_wel()
        if hfb:
            self._add_hfb()
        self._add_oc()
        if self.base.is_mf2005():
            self._add_pcg()

    def _spd_mf2005(self, results: pd.DataFrame, layers: List[int], option: Dict[str, Any]) -> List[Any]:
        all_results = []
        if isinstance(layers, list):
            botm = self.model.dis.botm.array.copy()
            top = self.model.dis.top.array.copy()
            for li, lay in enumerate(layers):
                for idx, row in results.iterrows():
                    if self.is_in_idomain(row):
                        if "depth" in results.columns:
                            if row.depth <= botm[lay - 1, row.row, row.col]:
                                botm[lay - 1, row.row, row.col] = row.depth - 0.1
                            if row["stage"] < botm[lay - 1, row.row, row.col]:
                                top[row.row, row.col] = row["stage"] + 0.1
                        all_results.append([lay - 1, row.row, row.col, *row[list(option.keys())]])
            for lay in range(self.model.modelgrid.nlay - 1, 0, -1):
                botm[lay] = np.where(botm[lay - 1] - botm[lay] <= 1, botm[lay - 1] - 1, botm[lay])
            self.model.dis.botm = botm
            self.model.dis.top = top
        else:
            for idx, row in results.iterrows():
                if self.is_in_idomain(row):
                    if "depth" in results.columns:
                        if self.model.modelgrid.botm[row.lay - 1][row.row, row.col] >= row["stage"] - row.depth:
                            self.model.modelgrid.botm[row.lay - 1][row.row, row.col] = row["stage"] - row.depth - 0.1
                        if row.lay == 1 and self.model.modelgrid.top[row.row, row.col] <= row["stage"] - row.depth:
                            self.model.modelgrid.top[row.row, row.col] = row["stage"] + 0.1
                    all_results.append([row.lay - 1, row.row, row.col, *row[list(option.keys())]])

        return all_results

    def _add_to_spd(self, row, option, lay, all_results):
        lay = int(lay)
        if self.is_structured():
            if self.is_in_idomain(row):
                all_results.append([(lay - 1, row.row, row.col), *row[list(option.keys())]])
        else:
            if self.is_in_idomain(row):
                all_results.append([(lay - 1, row.index_right), *row[list(option.keys())], row.name ])
        return all_results

    def _spd_mf6(self, results: pd.DataFrame, layers: List[int], option: Dict[str, Any]) -> List[Any]:
        all_results = []
        botm = self.model.dis.botm.array.copy()
        top = self.model.dis.top.array.copy()
        if isinstance(layers, list):
            for li, lay in enumerate(layers):
                for idx, row in results.iterrows():
                    if "depth" in results.columns:
                        if row.depth <= botm[lay - 1, row.index_right]:
                            botm[lay - 1, row.index_right] = row.depth - 0.1
                        if row["stage"] < botm[lay - 1, row.index_right]:
                            top[row.index_right] = row["stage"] + 0.1
                    all_results = self._add_to_spd(row, option, lay, all_results)
            for lay in range(1, self.model.modelgrid.nlay):
                botm[lay] = np.where(botm[lay - 1] - botm[lay] < 0.1, botm[lay - 1] - 0.1, botm[lay])

            self.model.dis.botm = botm
            self.model.dis.top = top
        else:
            for idx, row in results.iterrows():
                if "stage" in results.columns:
                    self.fix_riv(row)
                all_results = self._add_to_spd(row, option, row.lay, all_results)
        return all_results

    def _prepare_spd_df(self, option):
        geometry = option.pop("geometry")
        layers = option.pop("layers")
        results = self.process_geometry(geometry)
        if "bname" in option:
            results["bname"] = option.pop("bname")
        results = self.process_parameters(option, results)
        return results, layers

    def _add_source_sinks(self, package_name: str) -> Dict[int, List[Any]]:
        package_data = self.packages.get(package_name, {})
        step_std = {}
        for i, spd in package_data.items():
            all_results = []
            if spd is not None:
                if spd.get("add"):
                    std_ex = self.model.get_package(package_name.upper()).stress_period_data.data[i].tolist()
                    all_results.extend(std_ex)
                for option in spd["data"]:
                    results, layers = self._prepare_spd_df(option)
                    if self.base.is_mf6():
                        all_results.extend(self._spd_mf6(results, layers, option))
                    if self.base.is_mf2005():
                        all_results.extend(self._spd_mf2005(results, layers, option))
                step_std[i] = self._delete_duplicates(all_results) if package_name != 'wel' else self._sum_duplicates(
                    all_results)
            else:
                step_std[i] = {}
        if self.editing:
            self.model.remove_package(package_name)
        return step_std

    def _barrier_data(self, results, layers, option):
        all_results = []
        if isinstance(layers, list):
            for li, lay in enumerate(layers):
                for idx, row in results.iterrows():
                    all_results.append([(lay - 1, row.cellid1),
                                        (lay - 1, row.cellid2),
                                        *row[list(option.keys())]])
        return all_results

    @staticmethod
    def find_nearest_polygon(row, polygons_gdf):
        centroid_point = row['geometry']
        current_cellid = row['cellids']

        filtered_polygons = polygons_gdf[polygons_gdf.index != current_cellid]

        distances = filtered_polygons.geometry.distance(centroid_point)
        nearest_polygon_index = distances.idxmin()
        return nearest_polygon_index

    def intersects_centroid(self, line, gr):
        ix = gr.intersect(line.geometry, sort_by_cellid=False)
        lines_ix = gpd.GeoDataFrame(ix, geometry=ix.ixshapes)
        lines_ix = lines_ix.explode()
        lines_ix.geometry = lines_ix.centroid
        return lines_ix

    def get_nearest_poly(self, df):
        df["cellid1"] = df["cellids"]
        df['cellid2'] = df.apply(self.find_nearest_polygon, polygons_gdf=self.grid_poly, axis=1)
        return df

    def _add_barrier(self, package_name: str) -> Dict[int, List[Any]]:
        package_data = self.packages.get(package_name, {})
        step_std = {}
        for i, spd in package_data.items():
            all_results = []
            for option in spd["data"]:
                geometry = option.pop("geometry")
                layers = option.pop("layers")
                lines = gpd.read_file(geometry)
                gr = GridIntersect(self.model.modelgrid)
                for idx, line in lines.iterrows():
                    lines_ix = self.intersects_centroid(line, gr)
                    lines_ix = self.process_parameters(option, lines_ix)
                    lines_ix = self.get_nearest_poly(lines_ix)
                    if self.base.is_mf6():
                        all_results.extend(self._barrier_data(lines_ix, layers, option))
                    if self.base.is_mf2005():
                        pass

            step_std[i] = all_results
            if self.editing:
                self.model.remove_package(package_name)
        return step_std

    def _pars_raster(self, df, atr, atr_name):
        par_val = self.resample_raster(atr)
        if self.is_structured():
            df[atr_name] = par_val[df.row.values, df.col.values]
        else:
            df[atr_name] = par_val[df.index_right.values]
        return df

    def _pars_exact(self, df, atr, atr_name):
        if self.base.is_mf6():
            df = pd.merge(df, self.grid_poly, how="left", left_on="index_right", right_index=True)
        else:
            df = pd.merge(df, self.grid_poly, how="left", left_on=["row", "col"], right_on=["row", "col"])
        df['centroid'] = df.geometry_y.centroid

        with rasterio.open(atr) as src:
            def sample_raster(point):
                row, col = ~affine * (point.x, point.y)
                row, col = int(row), int(col)
                return band[col, row]

            affine = src.transform
            band = src.read(1)
            df[atr_name] = df['centroid'].apply(sample_raster)
            return df

    def _pars_top(self, df, atr_name, calc_val=0.):
        top = self.model.modelgrid.top
        if self.is_structured():
            df[atr_name] = top[df.row.values, df.col.values] + calc_val
        else:
            df[atr_name] = top[df.index_right.values] + calc_val
        return df

    def fix_riv(self, row):
        if not self.base.is_mf6():
            if self.model.modelgrid.botm[0][int(row["row"])][int(row["col"])] >= row["stage"] - row["depth"]:
                self.model.modelgrid.botm[0][int(row["row"])][int(row["col"])] = row["stage"] - row[
                    "depth"] - 0.5
        else:
            if self.model.modelgrid.botm[0][int(row["index_right"])] >= row["depth"]:
                self.model.modelgrid.botm[0][int(row["index_right"])] = row[
                    "depth"] - 0.5


    def process_parameters(self, params: Dict[str, Any], results: pd.DataFrame) -> pd.DataFrame:
        exact = params.pop("exact", None)
        alongline = params.pop("alongline", None)
        if alongline:
            results["stage"] = results.geometry_y.centroid.apply(lambda node: self.linear_interpolation(node, alongline))
            # FIXME: rewrite
            # self.model.modelgrid.top[zip(results["row"], results["col"])] = results["stage"]
            for idx, row in results.iterrows():
                self.fix_riv(row)

        for name, std_atr in params.items():
            if self.is_raster(std_atr):
                if exact:
                    results = self._pars_exact(results, std_atr, name)
                else:
                    results = self._pars_raster(results, std_atr, name)
            elif type(std_atr) is str and "top" in std_atr:
                calc_val = std_atr.replace("top", "").replace(" ", "")
                results = self._pars_top(results, name, calc_val=float(calc_val) if calc_val else 0)
            elif type(std_atr) is str and re.match(r"([a-zA-Z\s]+)([+-]\s*\d+)", std_atr):
                atr, calc_val = self.separate_string_and_number(std_atr)
                results[name] = results[atr] + float(calc_val)
            elif type(std_atr) is str and re.match(r"([a-zA-Z\s]+)([+-]\s*[a-zA-Z\s]+)", std_atr):
                atr1, atr2 = std_atr.split('-')
                results[name] = results[atr1.strip()] - results[atr2.strip()]
            elif self.is_numeric(std_atr):
                results[name] = std_atr
        return results

    def _sum_duplicates(self, results: List[Any]) -> List[Any]:
        sum_rate = {}
        for sublist in results:
            if self.base.is_mf6():
                key = sublist[0]
                value = float(sublist[1])
            else:
                key = (sublist[0], sublist[1], sublist[2])
                value = float(sublist[3])
            if sum_rate.get(key):
                sum_rate[key] += value
            else:
                sum_rate[key] = value
        if self.base.is_mf6():
            return [[key, sum_rate[key]] for key in sum_rate]
        else:
            return [[*key, sum_rate[key]] for key in sum_rate]

    def _delete_duplicates(self, results: List[Any]) -> List[Any]:
        unique_list = []
        seen_tuples = set()
        for sublist in results:
            if self.base.is_mf6():
                key = sublist[0]
            else:
                key = (sublist[0], sublist[1], sublist[2])
            if key not in seen_tuples:
                unique_list.append(sublist)
                seen_tuples.add(key)
        return unique_list

    def _add_chd(self):
        step_std = self._add_source_sinks("chd")
        if self.base.is_mf6():
            chd = flopy.mf6.ModflowGwfchd(self.model, stress_period_data=step_std, boundnames=True)
        if self.base.is_mf2005():
            chd = flopy.modflow.ModflowChd(self.model, stress_period_data=step_std)

    def _add_ghb(self):
        step_std = self._add_source_sinks("ghb")
        if self.base.is_mf6():
            ghb = flopy.mf6.ModflowGwfghb(self.model, stress_period_data=step_std, boundnames=True)
        else:
            ghb = flopy.modflow.ModflowGhb(self.model, stress_period_data=step_std)

    def _add_drn(self):
        step_std = self._add_source_sinks("drn")
        if self.base.is_mf6():
            drn = flopy.mf6.ModflowGwfdrn(self.model, stress_period_data=step_std, boundnames=True)
        else:
            drn = flopy.modflow.ModflowDrn(self.model, stress_period_data=step_std)

    def _add_wel(self):
        step_std = self._add_source_sinks("wel")
        if self.base.is_mf6():
            wel = flopy.mf6.ModflowGwfwel(self.model, stress_period_data=step_std)
        else:
            wel = flopy.modflow.ModflowWel(self.model, stress_period_data=step_std)

    def _add_riv(self):
        step_std = self._add_source_sinks("riv")
        if self.base.is_mf6():
            riv = flopy.mf6.ModflowGwfriv(self.model, stress_period_data=step_std, save_flows=True, boundnames=True)
        if self.base.is_mf2005():
            riv = flopy.modflow.ModflowRiv(self.model, stress_period_data=step_std)

    def _add_hfb(self):
        step_std = self._add_barrier("hfb")
        if self.base.is_mf6():
            hfb = flopy.mf6.ModflowGwfhfb(self.model, stress_period_data=step_std)
        else:
            hfb = None
            # hfb = flopy.modflow.ModflowHfb(self.model, hfb_data=step_std)

    def pnt_along_line(self, line):
        # FIXME: change distance_delta considering modelgrid space
        distance_delta = 0.1
        distances = np.arange(0, line.length, distance_delta)
        points = [line.interpolate(distance) for distance in distances]
        # multipoint = unary_union(points)
        return points

    def _add_oc(self):
        model_name = self.base.name
        if self.base.is_mf6():
            oc = flopy.mf6.ModflowGwfoc(
                self.base.model,
                pname="oc",
                budget_filerecord=f"{model_name}.cbb",
                head_filerecord=f"{model_name}.hds",
                headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
                saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
            )
        if self.base.is_mf2005():
            spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
            oc = flopy.modflow.ModflowOc(self.base.model, stress_period_data=spd, compact=True)

    def _add_pcg(self):
        pcg = flopy.modflow.ModflowPcg(self.base.model, mxiter=1000, hclose=0.0001, rclose=0.0001)

    @staticmethod
    def find_two_nearest_observations(node, gdf_obs):
        distances = gdf_obs.geometry.apply(lambda obs: node.distance(obs))
        nearest_two = distances.nsmallest(2).index
        return nearest_two, distances[nearest_two]

    # Function to perform linear interpolation between two nearest observation points
    def linear_interpolation(self, node, gdf_obs):
        # Get the two nearest observation points and their distances
        gdf_obs = gpd.read_file(gdf_obs)
        nearest_two, distances = self.find_two_nearest_observations(node, gdf_obs)

        # Extract the stage values and distances of the nearest points
        stage_values = gdf_obs.loc[nearest_two, 'z']
        dist_1, dist_2 = distances.iloc[0], distances.iloc[1]
        stage_1, stage_2 = stage_values.iloc[0], stage_values.iloc[1]

        # Perform linear interpolation
        if dist_1 + dist_2 > 0:
            interpolated_value = (stage_1 * dist_2 + stage_2 * dist_1) / (dist_1 + dist_2)
        else:
            # In case the two points are extremely close
            interpolated_value = stage_1
        return interpolated_value


class ModelObservations(GridHandler):
    def __init__(self, base: ModelBase, config: dict, editing):
        super().__init__(base)
        self.packages = config
        if editing:
            self.model.remove_package("OBS")
        self.add_packages()

    @staticmethod
    def _update_original_csv(option: dict, results: pd.DataFrame):
        dirname = os.path.dirname(option.get("geometry"))
        geometry = Path(option.get("geometry")).stem
        results.to_csv(os.path.join(dirname, f"{geometry}_text.csv"), index=False)

    def add_packages(self):
        if self.packages.get("wells"):
            self._add_wells_obs()

    def _add_wells_obs(self):
        options = self.packages.get("wells")
        obslist = []

        for option in options:
            results = self._process_well_option(option)
            obslist.extend(self._process_results(results, self.wel_obs_func))
        self._form_wellobs(obslist)

    def _form_wellobs(self, obslist):
        if self.base.is_mf6():
            obsdict = {}
            obsdict[f"{self.model.name}.obs.head.csv"] = obslist
            self._create_wellobs_package(obsdict)
        else:
            self._create_wellobs_package(obslist)

    def _process_well_option(self, option: dict) -> pd.DataFrame:
        results = self.process_geometry(option.get("geometry"))
        results = self._rename_or_merge(results, option)
        self._update_original_csv(option, results)
        return results

    def _rename_or_merge(self, results: pd.DataFrame, option: dict) -> pd.DataFrame:
        layers = option.get("layers")
        depth = option.get("depth")
        name = option.get("name")
        # if not name:
        #     results["name"] = "H" + (results.reset_index().index + 1).astype(str)
        # else:
        # results["name"] = "H" + results[name.lower()].astype(str)
        results["name"] = "H" + (results.reset_index().index + 1).astype(str)
        if not depth and layers:
            if type(layers) is str:
                results["lay"] = results[layers]
            else:
                results["lay"] = layers
        elif depth:
            results["depth"] = results[depth.lower()]
            results["lay"] = results.apply(self.get_lay_by_depth, axis=1,
                                           args=(self.model.modelgrid.top, self.model.modelgrid.botm))
        else:
            raise ValueError("Specify the layer number or depth=True")
        return results

    def _process_results(self, results: pd.DataFrame, wel_obs_func) -> List[Any]:
        obslist = []
        for _, row in results.iterrows():
            if self.is_in_idomain(row):
                obslist.append(wel_obs_func(row))
        return obslist

    def _create_wellobs_package(self, obsdict: Dict[str, List[Any]]):
        if self.base.is_mf6():
            flopy.mf6.ModflowUtlobs(
                self.model, print_input=True, continuous=obsdict
            )
        else:
            flopy.modflow.ModflowHob(self.model, obs_data=obsdict, iuhobsv=1)

    def wel_obs_func(self, row):
        if self.is_structured():
            if self.base.is_mf6():
                return [f"{row['name']}", "HEAD", (int(row["lay"]) - 1, row.row, row.col)]
            else:
                ts = [1, row["head"]]
                return flopy.modflow.HeadObservation(self.model, obsname=row["name"],
                                                     layer=int(row["lay"]) - 1, row=row.row, column=row.col,
                                                     time_series_data=ts)
        else:
            return [f"{row['name']}", "HEAD", (int(row["lay"]) - 1, row.index_right)]

    def get_lay_by_depth(self, row, top, botm):
        if row["depth"]:
            for i, bot in enumerate(botm):
                if self.is_structured():
                    if row["depth"] <= top[(int(row["row"]), int(row["col"]))] - bot[
                        (int(row["row"]), int(row["col"]))]:
                        return i + 1
                else:
                    if row["depth"] <= top[int(row["index_right"])] - bot[int(row["index_right"])]:
                        return i + 1
        return len(botm)


class ModelBuilder:
    def __init__(self, config: dict, external=False, editing=False):
        self.config = config
        self.editing = editing
        self.external = external
        self._initialize_config()
        self.base = ModelBase(config.get("base"), editing)
        grid = config.get("grid")
        if grid.get("type") == "structured":
            self.grid = MfStructuredGrid(self.base, grid, editing)
        else:
            self.grid = MfUnstructuredGrid(self.base, grid, editing)
        self.parameters = ModelParameters(self.base, config.get("parameters"), editing) if config.get(
            "parameters") else None
        self.sources = ModelSourcesSinks(self.base, config.get("sources"), editing) if config.get(
            "sources") else None
        self.observations = ModelObservations(self.base, config.get("observations"), editing) if config.get(
            "observations") else None

    def _initialize_config(self):
        validator = ConfigValidator(self.config, self.editing)
        validator.validate_config()

    def run_mf6(self):
        if self.external:
            self.base.simulation.set_all_data_external(True)
        self.base.simulation.write_simulation()
        return self.base.simulation.run_simulation()

    def run_mf2005(self):
        self.base.model.write_input()
        self.base.model.check()
        return self.base.model.run_model()

    def run(self):
        if self.base.is_mf6():
            success, buff = self.run_mf6()
        else:
            success, buff = self.run_mf2005()
        if success:
            for line in buff:
                print(line)
        else:
            raise ValueError("Failed to run.")
