import os
import stat
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from plpygis import Geometry
from shapely import MultiPoint, Point
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
        self.units = config.get("units", "DAYS")
        self.simulation, self.model = self.__init_model()

    def __init_model(self):
        sim = flopy.mf6.MFSimulation(
            sim_name=self.name, exe_name=self.exe, version="mf6", sim_ws=self.workspace
        )
        if self.steady:
            tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
                sim, pname="tdis", time_units=self.units, nper=1, perioddata=[(1.0, 1, 1.0)]
            )
        gwf = flopy.mf6.ModflowGwf(sim, modelname=self.name, model_nam_file=f"{self.name}.nam")
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")
        return sim, gwf


class ModelAbstract:
    def __init__(self, model):
        self.model = model

    def raster_resample(self, path, method="nearest"):
        rio = Raster.load(os.path.join(self.model.model_ws, path))
        data = rio.resample_to_grid(self.model.modelgrid, band=rio.bands[0], method=method)
        return data

    def _intersection_grid(self, geometry, attribute="zone"):
        ix = GridIntersect(self.model.modelgrid, method="vertex")
        result = []
        num = []
        if type(geometry) is str:
            layer = gpd.read_file(os.path.join(self.model.model_ws, geometry))
            geometry = layer.geometry

        for n, row in enumerate(geometry):
            # FIXME it's better to use intersect method instead of intersects
            cells = ix.intersects(row)
            result.extend(cells)
            num.extend([n] * len(cells))
        return result, num


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
            return self.top
        else:
            return self.raster_resample(self.top)

    def _prepare_botm(self):
        for i, bot in enumerate(self.botm):
            if type(bot) is str:
                self.botm[i] = self.raster_resample(bot)
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
                top=self._prepare_top(),
                botm=self._prepare_botm(),
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
            dem_data = self._prepare_top()
            bot_data = self._prepare_botm()
            bot_data = np.where(dem_data - bot_data < 7, dem_data - 7, bot_data)
            disv.top = dem_data
            disv.botm = bot_data


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
                result, _ = self._intersection_grid(poly[0])
                rch_std.append([[(0, *node[0]), poly[1]] for node in result])
            flopy.mf6.ModflowGwfrch(self.model, stress_period_data=[item for sublist in rch_std for item in sublist])
        elif type(rch) is float:
            flopy.mf6.ModflowGwfrcha(self.model, recharge={0: rch})
        elif type(rch) is str:
            rch = self.raster_resample(rch)
            flopy.mf6.ModflowGwfrcha(self.model, recharge={0: rch})

    def _add_ic(self):
        options = self.packages.get("ic")
        ic = flopy.mf6.ModflowGwfic(self.model, strt=options.get("head") if options else self.model.modelgrid.top)

    def add_packages(self):
        self._add_ic()
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
        options = self.packages.get("riv")
        std = []
        last_num = 0
        for option in options:
            result, num = self._intersection_grid(option.get("geometry"))
            if option["stage"] == "top":
                std.append([[
                    (0, *cell[0]) if type(self.model.modelgrid) is fgrid.StructuredGrid else (0, cell[0],),
                    self.model.modelgrid.top[cell[0]], option["cond"],
                    self.model.modelgrid.top[cell[0]] - option["depth"], f"zn_{zn + last_num}"] for cell, zn in
                    zip(result, num)]
                )
                last_num = max(num) + 1
        std = [item for sublist in std for item in sublist] if len(std) > 1 else std[0]
        riv = flopy.mf6.ModflowGwfriv(self.model, stress_period_data=std, save_flows=True, boundnames=True)

    def _add_ghb(self):
        options = self.packages.get("ghb")
        std = []
        for option in options:
            result, _ = self._intersection_grid(option.get("geometry"))
            std.append(
                [[(lay, *cell[0]) if type(self.model.modelgrid) is fgrid.StructuredGrid else (lay, cell[0],),
                  option["head"], option["cond"]] for lay in option["layers"] for cell in result]
            )
        std = [item for sublist in std for item in sublist] if len(std) > 1 else std[0]
        ghb = flopy.mf6.ModflowGwfghb(self.model, stress_period_data=std)

    def _add_drn(self):
        options = self.packages.get("drn")
        std = []
        for option in options:
            if option.get("geometry") == "all":
                cells = self.model.modelgrid.cell2d
            else:
                cells, _ = self._intersection_grid(option.get("geometry"))
            heads = self.model.modelgrid.top if option.get("head") == "top" else option.get("head")
            result = zip(cells, heads)

            std.append(
                [[(lay, *cell[0]) if type(self.model.modelgrid) is fgrid.StructuredGrid else (lay, cell[0],),
                  top, option["cond"]] for lay in option["layers"] for cell, top in result]
            )
        std = [item for sublist in std for item in sublist] if len(std) > 1 else std[0]
        drn = flopy.mf6.ModflowGwfdrn(self.model, stress_period_data=std)

    def add_packages(self):
        if self.packages.get("riv"):
            self._add_river()
        if self.packages.get("ghb"):
            self._add_ghb()
        if self.packages.get("drn"):
            self._add_drn()


class ModelObservations:
    def __init__(self, model, config: dict):
        self.model = model
        self.packages = config
        self.add_packages()

    def _add_wells_obs(self):
        options = self.packages.get("wells")
        obslist = []
        obsdict = {}
        if options.endswith(".csv"):
            obs = pd.read_csv(os.path.join(self.model.model_ws, options))
            mp = MultiPoint(points=[Point(x, y, ids) for x, y, ids in zip(obs.X, obs.Y, obs.id1)])
            ix = GridIntersect(self.model.modelgrid, method="vertex")
            nodes = ix.intersect(mp)
            obslist = [(f"h_{n[1].z}", "HEAD", (0, n[0]))
                       if type(n[1]) is Point else
                       (f"h_{n[1].geoms[0].z}", "HEAD", (0, n[0])) for n in nodes]
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
