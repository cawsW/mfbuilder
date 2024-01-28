import os
import stat
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from plpygis import Geometry
from shapely import Polygon
import flopy
import flopy.discretization as fgrid
from flopy.utils.gridgen import Gridgen
from flopy.utils import GridIntersect, Raster
from shapely.ops import unary_union

import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata


class ModelBase:
    def __init__(self, config: dict, editing):
        if config.get("name"):
            self.name = config.get("name")
        else:
            raise ValueError("No name specified")

        if config.get("workspace"):
            self.workspace = config.get("workspace")
            if not os.path.exists(self.workspace):
                os.mkdir(self.workspace)
        else:
            raise ValueError("No workspace specified")

        if config.get("exe"):
            self.exe = config.get("exe")
            if not os.path.exists(self.exe):
                raise ValueError(f"Executable {self.exe} does not exist")
            else:
                st = os.stat(self.exe)
                os.chmod(self.exe, st.st_mode | stat.S_IEXEC)

        self.version = config.get("version", "mf6")
        self.steady = config.get("steady", True)
        self.perioddata = config.get("perioddata")
        self.units = config.get("units", "DAYS")
        self.editing = editing
        self.simulation, self.model = self.__init_model()

    def __init_model(self):
        if not self.editing:
            sim = flopy.mf6.MFSimulation(
                sim_name=self.name, exe_name=self.exe, version="mf6", sim_ws=self.workspace
            )
            if self.steady:
                tdis = flopy.mf6.ModflowTdis(
                    sim, pname="tdis", time_units=self.units, nper=1, perioddata=[(1.0, 1, 1.0)]
                )
            else:
                if self.perioddata:
                    tdis = flopy.mf6.ModflowTdis(
                        sim, pname="tdis", time_units=self.units, nper=len(self.perioddata), perioddata=self.perioddata
                    )
                else:
                    raise ValueError("No time steps specified for transient model")
            gwf = flopy.mf6.ModflowGwf(
                sim, modelname=self.name, model_nam_file=f"{self.name}.nam",
                save_flows=True,
                newtonoptions="NEWTON UNDER_RELAXATION",
            )
            ims = flopy.mf6.modflow.mfims.ModflowIms(sim,
                                                     pname="ims",
                                                     complexity="SIMPLE",
                                                     linear_acceleration="BICGSTAB",
                                                     )
        else:
            sim = flopy.mf6.MFSimulation.load(sim_ws=self.workspace, exe_name=self.exe)
            if not self.steady:
                if self.perioddata:
                    sim.remove_package("tdis")
                    tdis = flopy.mf6.ModflowTdis(
                        sim, pname="tdis", time_units=self.units, nper=len(self.perioddata), perioddata=self.perioddata
                    )
            gwf = sim.get_model(self.name)
        return sim, gwf


class ModelAbstract:
    def __init__(self, model):
        self.model = model

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
        rio = Raster.load(os.path.join(self.model.model_ws, path))
        data = rio.resample_to_grid(self.model.modelgrid, band=rio.bands[0], method=method)
        return data

    def _intersection_grid_attrs(self, geometry, attribute=[]):
        if type(geometry) is str:
            layer = gpd.read_file(os.path.join(self.model.model_ws, geometry))
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
                    st = os.stat(self.gridgen_exe)
                    os.chmod(self.gridgen_exe, st.st_mode | stat.S_IEXEC)
                else:
                    raise ValueError("No gridgen executable specified")
                self.gridgen_ws = config.get("gridgen_workspace", os.path.join(self.model.model_ws, "grid"))
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
        if config.get("botm"):
            self.botm = config.get("botm")
            if len(self.botm) != self.nlay:
                raise ValueError("Number of botm layers does not match number of layers")
        else:
            raise ValueError("No bottom specified")
        self.buffer = config.get("buffer", False)
        self.xmin, self.ymin, self.xmax, self.ymax = self.__init_bounds()
        self.min_thickness = config.get("min_thickness", 0.1)
        if not editing:
            self.create_dis()

    def check_boundary(self, boundary):
        if type(boundary) is str:
            boundary = gpd.read_file(os.path.join(self.model.model_ws, boundary), crs=self.proj_string)
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
            prj=self.proj_string, nlay=self.nlay
        )
        ix = GridIntersect(sgr, method="vertex")
        if type(self.boundary) is gpd.GeoDataFrame:
            result = ix.intersects(self.boundary.geometry[0])
        else:
            result = ix.intersects(self.boundary)
        idomain = np.zeros((self.nlay, n_col, n_row), dtype=np.int)
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
                layer = gpd.read_file(os.path.join(self.model.model_ws, data), crs=self.proj_string)
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
        if self.typegrd == "structured":
            n_row, n_col, idomain = self._clip_structure()
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
        if not all(isinstance(x, float) for x in self.botm):
            idomain, elevations, dem = self.form_idomain()
            disv.idomain = idomain
            disv.top = dem
            disv.botm = elevations
        else:
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
                    bot_data[i] = bot_data[i-1] - bot["thick"]
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

    def _add_sto(self):
        options = self.packages.get("sto")
        steady_state = {step: True for step in options.get("steady_state")}
        transient = {step: True for step in options.get("transient")}
        flopy.mf6.ModflowGwfsto(self.model, steady_state=steady_state, transient=transient,
                                sy=options.get("sy"), ss=options.get("ss"), iconvert=options.get("iconvert"))

    def _add_ic(self):
        options = self.packages.get("ic")
        ic = flopy.mf6.ModflowGwfic(self.model, strt=options.get("head") if options else self.model.modelgrid.top)

    def add_packages(self):
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


class ModelSourcesSinks(ModelAbstract):
    def __init__(self, model, config: dict, editing):
        super().__init__(model)
        self.packages = config
        self.editing = editing
        self.add_packages()

    def _add_river(self):
        data = self.packages.get("riv")

        def river_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(int(row[args[1]]), int(row[args[0]]))] == 1:
                    if stages == "top":
                        top = self.model.modelgrid.top[(int(row[args[1]]), int(row[args[0]]))]
                    else:
                        top = stages[(int(row[args[1]]), int(row[args[0]]))]
                    return [(lay - 1, int(row[args[1]]), int(row[args[0]])), top, row["cond"], top - option["depth"],
                            f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']
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
            step_std[i] = all_results
        return step_std

    def _add_ghb(self):
        data = self.packages.get("ghb")

        def ghb_func(row, *args):
            if len(args) == 2:
                return [(lay - 1, int(row[args[1]]), int(row[args[0]])), row["head"], option["cond"]]
            else:
                return [(lay - 1, int(row[args[0]])), row["head"], option["cond"]]

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
                if not cond:
                    results["head"] = option.get("head")
                for lay in option["layers"]:
                    all_results.extend(self._process_results(results, ghb_func))
            step_std[i] = all_results

        return step_std

    def _add_drn(self):
        data = self.packages.get("drn")

        def drn_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(int(row[args[1]]), int(row[args[0]]))] == 1:
                    head = self.model.modelgrid.top[(row[args[1]], row[args[0]])]
                    return [(lay - 1, int(row[args[1]]), int(row[args[0]])), head, option["cond"],
                            f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']
            else:
                head = self.model.modelgrid.top[row[args[0]]]
                return [(lay - 1, row[args[0]]), head, option["cond"],
                        f'{option["bname"]}_{lay}{row.name}' if option.get("bname") else f'{lay}{row.name}']

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                if option.get("geometry") == "all":
                    if type(self.model.modelgrid) is fgrid.StructuredGrid:
                        results = pd.DataFrame({"row": [i for i in range(self.model.modelgrid.ncol) for j in range(self.model.modelgrid.nrow)],
                                               "col": [j for i in range(self.model.modelgrid.ncol) for j in range(self.model.modelgrid.nrow)]})
                    else:
                        results = pd.DataFrame({"index_right": [cell[0] for cell in self.model.modelgrid.cell2d]})
                else:
                    results = self._intersection_grid_attrs(option.get("geometry"))
                for lay in option["layers"]:
                    all_results.extend(self._process_results(results, drn_func))
            step_std[i] = all_results

        return step_std

    def _add_chd(self):
        data = self.packages.get("chd")

        def chd_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(int(row[args[1]]), int(row[args[0]]))] == 1:
                    if type(heads) is not int or type(heads) is not float:
                        head = heads[(row[args[1]], row[args[0]])]
                    else:
                        head = heads
                    return [(lay - 1, int(row[args[1]]), int(row[args[0]])), head]
            else:
                if type(heads) is not int or type(heads) is not float:
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

    def pnt_along_line(self, line):
        # FIXME: change distance_delta considering modelgrid space
        distance_delta = 0.1
        distances = np.arange(0, line.length, distance_delta)
        points = [line.interpolate(distance) for distance in distances] + [line.boundary.geoms[1]]
        # multipoint = unary_union(points)
        return points

    @staticmethod
    def delete_duplicates_grid_package(example_list):
        unique_list = []
        seen_tuples = set()
        for sublist in example_list:
            if sublist[0] not in seen_tuples:
                unique_list.append(sublist)
                seen_tuples.add(sublist[0])
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
                ln = gpd.read_file(os.path.join(self.model.model_ws, option.get("geometry")))
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
            step_std = self._add_river()
            riv = flopy.mf6.ModflowGwfriv(self.model, stress_period_data=step_std, save_flows=True, boundnames=True)
        if ghb:
            if self.editing:
                self.model.remove_package("GHB")
            step_std = self._add_ghb()
            ghb = flopy.mf6.ModflowGwfghb(self.model, stress_period_data=step_std)
        if drn:
            if self.editing:
                self.model.remove_package("DRN")
            step_std = self._add_drn()
            drn = flopy.mf6.ModflowGwfdrn(self.model, stress_period_data=step_std, boundnames=True)
        if chd:
            if self.editing:
                self.model.remove_package("CHD")
            step_std = self._add_chd()
            chd = flopy.mf6.ModflowGwfchd(self.model, stress_period_data=step_std)
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
        if not editing:
            self.add_packages()

    def _add_wells_obs(self):
        options = self.packages.get("wells")

        def wel_obs_func(row, *args):
            if len(args) == 2:
                if self.model.modelgrid.idomain[0][(row[args[1]], row[args[0]])] == 1:
                    return [(f"h_{row['name']}", "HEAD", (row["lay"] - 1, row[args[1]], row[args[0]]))]
            else:
                return [(f"h_{row['name']}", "HEAD", (row["lay"] - 1, row[args[0]]))]

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
                obs = pd.read_csv(os.path.join(self.model.model_ws, option.get("data")))
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
                    results_mod[list(obs.columns) + ["lay"]].to_csv(os.path.join(self.model.model_ws, option.get("data")), index=False)
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
            self.grid = ModelGrid(self.base.model, config.get("grid"), editing)
        else:
            raise ValueError("No grid options specified")
        if config.get("parameters"):
            self.parameters = ModelParameters(self.base.model, config.get("parameters"), editing)
        else:
            print("No parameters options specified")
        self.sources = ModelSourcesSinks(self.base.model, config.get("sources"), editing) if config.get("sources") else None
        self.observations = ModelObservations(self.base.model, config.get("observations"), editing) if config.get(
            "observations") else None
        self.external = external

    def output_package(self):
        model_name = self.base.name
        oc = flopy.mf6.ModflowGwfoc(
            self.base.model,
            pname="oc",
            budget_filerecord=f"{model_name}.cbb",
            head_filerecord=f"{model_name}.hds",
            headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
            printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )

    def run(self):
        self.output_package()
        if self.external:
            self.base.simulation.set_all_data_external(True)
        self.base.simulation.write_simulation()
        success, buff = self.base.simulation.run_simulation()
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
        interpolated_data = np.where(interpolated_data > 1e6, -999, interpolated_data)
        new_dataset = rasterio.open(
         os.path.join(dir, f"{name}.tif"),
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

    def _create_dir(self):
        out_dir = os.path.join(self.base.model.model_ws, "output")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir

    def get_observations(self, directory):
        well_heads = self.observations.packages.get("wells")
        if well_heads:
            obs = pd.read_csv(os.path.join(self.base.model.model_ws, well_heads[0].get("data")))
            obs["name"] = "H_" + obs["name"].astype(str)
            model_obs = pd.read_csv(os.path.join(self.base.model.model_ws, f"{self.base.model.name}.obs.head.csv"))
            model_obs = model_obs.transpose().reset_index()
            model_obs.columns = ["name", "head_model"]
            res = model_obs.merge(obs, on="name")
            res["residual"] = res["head"] - res["head_model"]
            res.to_csv(os.path.join(directory, "residuals.csv"), index=False)

    def export(self):
        dir = self._create_dir()
        perioddata = self.base.simulation.get_package("tdis").perioddata.array
        ex_arr = []
        surf_arr = []
        ex_arr.extend([(f"head_{i}{step}_lay_", self.get_heads((step, i))) for i, per in enumerate(perioddata) for step in range(per[1])])
        ex_arr.extend([(f"npf_lay_", self.get_npf())])
        rch = self.get_rch()
        if rch is not None:
            ex_arr.extend([(f"rch_lay_", rch)])
        surf_arr.extend([(f"bot_lay_", self.base.model.modelgrid.botm)])
        surf_arr.extend([(f"top_lay_", [self.base.model.modelgrid.top])])
        for name, arr in ex_arr:
            for lay, la in enumerate(arr):
                self.create_raster(50, la, f"{name}{lay}", dir)

        for name, arr in surf_arr:
            for lay, la in enumerate(arr):
                self.create_raster(50, la, f"{name}{lay}", dir)

        self.get_observations(dir)
