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


class ModelBase:
    def __init__(self, config: dict):
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
        self.simulation, self.model = self.__init_model()

    def __init_model(self):
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
        gwf = flopy.mf6.ModflowGwf(sim, modelname=self.name, model_nam_file=f"{self.name}.nam")
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")
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
                        (a[0][i], a[1][j]),
                        (a[0][i + 1], a[1][j]),
                        (a[0][i + 1], a[1][j + 1]),
                        (a[0][i], a[1][j + 1]),
                        (a[0][i], a[1][j]),
                    ])
                    polygons.append(poly)
                    rowcol.append((i, j))
        else:
            polygons = [Polygon(array.vertices) for array in poly_arr]
        griddf = gpd.GeoDataFrame(data=rowcol, columns=["row", "col"], geometry=polygons)
        griddf = griddf.replace(np.nan, -999)
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
        join_pdf = join_pdf.astype({"row": int, "col": int, "index_right": int})
        return join_pdf[["index_right", "row", "col"] + attribute]

    def _process_results(self, results, process_func):
        std = []
        for idx, row in results.iterrows():
            if type(self.model.modelgrid) is fgrid.StructuredGrid:
                std.append(process_func(row, "row", "col"))
            else:
                std.append(process_func(row, "index_right"))
        return std


class ModelGrid(ModelAbstract):
    def __init__(self, model, config: dict):
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
            else:
                self.botm[i] = bot * np.ones(self.model.modelgrid.ncpl)
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
                xorigin=self.xmin, yorigin=self.xmin, angrot=0
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

        idomain, elevations, dem = self.form_idomain()
        disv.idomain = idomain
        disv.top = dem
        disv.botm = elevations

    def form_idomain(self):
        elevations, dem = self._reduce_geology()
        dom = np.array([dem, *elevations])
        idomain = []
        for i, el in enumerate(dom[1:]):
            idomain.append(np.where(dom[i] - el == 0, -1, 1))
        return idomain, elevations, dem

    def _reduce_geology(self):
        dem_data = self._prepare_top()
        bot_data = self._prepare_botm()
        bot_data[0] = np.where(dem_data - bot_data[0] < self.min_thickness, dem_data - self.min_thickness, bot_data[0])
        elevations = np.array([dem_data, *bot_data])[::-1]
        prepare_elev = []
        for lay in range(self.nlay):
            prepare_elev.append(np.minimum.reduce(elevations[lay:]))
        return prepare_elev[::-1], dem_data


class ModelParameters(ModelAbstract):
    def __init__(self, model, config: dict):
        super().__init__(model)
        self.packages = config
        self.add_packages()

    def _add_npf(self):
        npf = self.packages.get("npf")
        hk = npf.get("k")
        icell = npf.get("icelltype")
        for i, k in enumerate(hk):
            if type(k) is str:
                hk[i] = self.raster_resample(k)
        npf = flopy.mf6.ModflowGwfnpf(self.model, save_flows=True, k=hk, icelltype=icell)

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
            self._add_sto()
        if self.packages.get("npf"):
            self._add_npf()
        if self.packages.get("rch"):
            self._add_rch()


class ModelSourcesSinks(ModelAbstract):
    def __init__(self, model, config: dict):
        super().__init__(model)
        self.packages = config
        self.add_packages()

    def _add_river(self):
        data = self.packages.get("riv")

        def river_func(row, *args):
            if len(args) == 2:
                top = self.model.modelgrid.top[(row[args[0]], row[args[1]])]
                return [(0, row[args[0]], row[args[1]]), top, option["cond"], top - option["depth"]]
            else:
                top = self.model.modelgrid.top[row[args[0]]]
                return [(0, row[args[0]]), top, option["cond"], top - option["depth"]]

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                results = self._intersection_grid_attrs(option.get("geometry"))
                if option["stage"] == "top":
                    all_results.extend(self._process_results(results, river_func))
            step_std[i] = all_results

        return step_std

    def _add_ghb(self):
        data = self.packages.get("ghb")

        def ghb_func(row, *args):
            if len(args) == 2:
                return [(lay - 1, row[args[0]], row[args[1]]), option["head"], option["cond"]]
            else:
                return [(lay - 1, row[args[0]]), option["head"], option["cond"]]

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                results = self._intersection_grid_attrs(option.get("geometry"))
                for lay in option["layers"]:
                    all_results.extend(self._process_results(results, ghb_func))
            step_std[i] = all_results

        return step_std

    def _add_drn(self):
        data = self.packages.get("drn")

        def drn_func(row, *args):
            if len(args) == 2:
                head = self.model.modelgrid.top[(row[args[0]], row[args[1]])]
                return [(lay - 1, row[args[0]], row[args[1]]), head, option["cond"]]
            else:
                head = self.model.modelgrid.top[row[args[0]]]
                return [(lay - 1, row[args[0]]), head, option["cond"]]

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                if option.get("geometry") == "all":
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
                return [(lay - 1, row[args[0]], row[args[1]]), option["head"]]
            else:
                return [(lay - 1, row[args[0]]), option["head"]]

        step_std = {}
        for i, options in data.items():
            all_results = []
            for option in options:
                results = self._intersection_grid_attrs(option.get("geometry"))
                for lay in option["layers"]:
                    all_results.extend(self._process_results(results, chd_func))
            step_std[i] = all_results

        return step_std

    def _add_wel(self):
        data = self.packages.get("wel")

        def wel_func(row, *args):
            if len(args) == 2:
                return [(int(row["lay"]) - 1, row[args[0]], row[args[1]]), row["q"]]
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

    def add_packages(self):
        riv = self.packages.get("riv")
        ghb = self.packages.get("ghb")
        drn = self.packages.get("drn")
        chd = self.packages.get("chd")
        wel = self.packages.get("wel")

        if riv:
            step_std = self._add_river()
            riv = flopy.mf6.ModflowGwfriv(self.model, stress_period_data=step_std, save_flows=True, boundnames=True)
        if ghb:
            step_std = self._add_ghb()
            ghb = flopy.mf6.ModflowGwfghb(self.model, stress_period_data=step_std)
        if drn:
            step_std = self._add_drn()
            drn = flopy.mf6.ModflowGwfdrn(self.model, stress_period_data=step_std)
        if chd:
            step_std = self._add_chd()
            chd = flopy.mf6.ModflowGwfchd(self.model, stress_period_data=step_std)
        if wel:
            step_std = self._add_wel()
            wel = flopy.mf6.ModflowGwfwel(self.model, stress_period_data=step_std)


class ModelObservations(ModelAbstract):
    def __init__(self, model, config: dict):
        super().__init__(model)
        self.packages = config
        self.add_packages()

    def _add_wells_obs(self):
        options = self.packages.get("wells")

        def wel_obs_func(row, *args):
            if len(args) == 2:
                return [(f"h_{row['name']}", "HEAD", (row["lay"] - 1, row[args[0]], row[args[1]]))]
            else:
                return [(f"h_{row['name']}", "HEAD", (row["lay"] - 1, row[args[0]]))]

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
                obs = pd.read_csv(os.path.join(self.model.model_ws, option.get("data")))
                results = self._intersection_grid_attrs(obs, attribute=attributes)
                if not name:
                    results["name"] = option.get("name")
                else:
                    results.rename(columns={name: 'name'}, inplace=True)
                if not lay:
                    results["lay"] = option.get("layers")
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
    def __init__(self, config: dict, external=False):
        if config.get("base"):
            self.base = ModelBase(config.get("base"))
        else:
            raise ValueError("No base options specified")

        if config.get("grid"):
            self.grid = ModelGrid(self.base.model, config.get("grid"))
        else:
            raise ValueError("No grid options specified")

        if config.get("parameters"):
            self.parameters = ModelParameters(self.base.model, config.get("parameters"))
        else:
            raise ValueError("No parameters options specified")

        self.sources = ModelSourcesSinks(self.base.model, config.get("sources")) if config.get("sources") else None
        self.observations = ModelObservations(self.base.model, config.get("observations")) if config.get(
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
