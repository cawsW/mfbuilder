import os
import shutil
from pathlib import Path

import numpy as np
import pyemu
import flopy
from flopy.utils import GridIntersect
from flopy.export.shapefile_utils import shp2recarray


class PstBuilder:
    def __init__(self, config, read=True, onlycreate=True):
        self.clb_model_ws = "calibration"
        self.tmp_model_ws = "pestrun"
        self.base = config.get("base")
        self.ws = self.base.get("workspace")
        self.modelname = self.base.get("modelname")
        self.pestname = f"{self.base.get("pestname")}.pst"
        self.pest_version = self.base.get("version")
        self.model_version = self.base.get("version_model")
        self.observations = config.get("observations")
        self.parameters = config.get("parameters")
        self.helpers = config.get("helpers")
        self.sim, self.model = self._get_model(self.ws)
        self.grid, self.nlay = self.model.modelgrid, self.model.modelgrid.nlay
        self.onlycreate = onlycreate
        self.external = self.base.get("external")
        if not read:
            self.pf = self._create_pstfrom()

    @staticmethod
    def _move_model(fromdir, todir):
        if os.path.exists(todir):
            shutil.rmtree(todir)
        shutil.copytree(fromdir, todir)

    def _get_model(self, ws):
        if self.model_version == "mf6":
            sim = flopy.mf6.MFSimulation.load(sim_ws=ws)
            m = sim.get_model(self.modelname)
        else:
            sim = None
            m = flopy.modflow.Modflow.load(f"{self.modelname}.nam", model_ws=ws)
        return sim, m

    def _change_file_structure(self):
        txtfiles = []
        for par, calpars in self.parameters.items():
            for configs in calpars:
                typef = configs.get("type")
                if typef == "pilotpoint":
                    txtfiles.extend(self._get_par_filename(par))
        for filet in txtfiles:
            with open(os.path.join(self.clb_model_ws, filet), 'r') as file:
                lines = file.readlines()
            numbers = []
            for line in lines:
                words = line.split()
                for word in words:
                    number = float(word)
                    numbers.append(number)
            np.savetxt(os.path.join(self.clb_model_ws, filet), np.array(numbers))

    def _create_pstfrom(self):
        self._move_model(self.ws, self.clb_model_ws)
        # self._change_file_structure()
        return pyemu.utils.PstFrom(original_d=self.clb_model_ws, new_d=self.tmp_model_ws,
                                   remove_existing=True, longnames=False, spatial_reference=self.grid,
                                   zero_based=False)

    def _add_observations(self):
        for key, pars in self.observations.items():
            filename = pars.pop("filename")
            name = pars.pop("name")
            hds = self.pf.add_observations(filename, insfile=f"{filename}.ins",
                                           prefix=name, obsgp=f"{name}_gp",
                                           **pars,
                                           )

    def _form_geostats(self, geocnf):
        model = geocnf.get("model")
        if model == "exponential":
            v = pyemu.geostats.ExpVario(a=geocnf.get("a"), contribution=geocnf.get("contribution"))
        else:
            v = None
        return pyemu.geostats.GeoStruct(variograms=v, transform="log")

    def _get_par_filename(self, par):
        path = self.ws
        if self.external:
            path = os.path.join(self.ws, self.external)
        files = os.listdir(path)
        parfiles = []
        for file in files:
            if self.model_version == "mf6" and par == "k":
                par = "npf_k"
            if f"{par}" in file and file.endswith("txt"):
                parfiles.append(file)
            if self.model_version == "mf2005" and f"{par}" in file:
                parfiles.append(file)
        return parfiles

    def _add_par_pp(self, name, configs, files):
        pp_name = {}
        geocnf = configs.pop("geostats")
        if geocnf:
            geostruct = self._form_geostats(geocnf)
        else:
            geostruct = None
        for i, file in enumerate(files):
            pp_name[f"{name}_{i}_inst0pp.dat.tpl"] = geostruct
            path = file
            if self.external:
                path = os.path.join(self.external / file)
            self.pf.add_parameters(filenames=path, par_type="pilotpoint",
                                   par_name_base=f"{name}_{i}", pargp=f"{name}_{i}_gp",
                                   geostruct=geostruct,
                                   **configs)
        return pp_name

    def _add_par_cnst(self, name, configs, files):
        for i, file in enumerate(files):
            self.pf.add_parameters(filenames=file, par_type="constant",
                                   par_name_base=f"{name}_{i}", pargp=f"{name}_{i}_gp",
                                   **configs)

    def _get_domain(self):
        return np.array(self.grid.idomain)

    def _get_zones(self, shp):
        idomain = self._get_domain()
        ix = GridIntersect(self.grid, method="vertex")
        ra = shp2recarray(shp)
        geoms = [x[1] for x in ra]
        zone = 1
        for geo in geoms:
            cells = ix.intersects(geo, shapetype="linestring")
            for cell in cells:
                idomain[0][cell[0]] = zone
            zone += 1
        return idomain

    def _add_par_zone(self, name, configs, files):
        zones = self._get_zones(configs.pop("zones"))
        # zones = np.array([zone.flatten() for zone in zones])
        for i, file in enumerate(files):
            self.pf.add_parameters(filenames=file, par_type="zone",
                                   zone_array=zones,
                                   par_name_base=f"{name}_{i}", pargp=f"{name}_{i}_gp",
                                   **configs)

    def _add_parameters(self):
        pp_name = {}
        for par, calpars in self.parameters.items():
            for configs in calpars:
                typef = configs.pop("type")
                filename = configs.pop("filename", None)
                if filename:
                    files = [filename]
                else:
                    files = self._get_par_filename(par)
                if typef == "pilotpoint":
                    pp_par = self._add_par_pp(par, configs, files)
                    if pp_par:
                        pp_name.update(pp_par)
                if typef == "constant":
                    self._add_par_cnst(par, configs, files)
                if typef == "zone":
                    self._add_par_zone(par, configs, files)
        return pp_name

    def _prepare_run(self):
        self.pf.mod_sys_cmds.append(f"{self.model_version}.exe {self.modelname}.nam")
        if self.helpers:
            for fnc in self.helpers:
                self.pf.add_py_function("helpers.py", fnc.get("func"), is_pre_cmd=fnc.get("pre"))

    def create_pst(self):
        pst = self.pf.build_pst(self.pestname, version=1)
        pst.write(os.path.join(self.tmp_model_ws, self.pestname))
        pyemu.os_utils.run(f"{self.pest_version} {self.pestname}", cwd=self.tmp_model_ws)
        return pst

    def _add_first_regularization(self, pst, pp_name):
        for k, v in pp_name.items():
            df = pyemu.pp_utils.pp_tpl_to_dataframe(os.path.join(self.tmp_model_ws, k))
            cov = v.covariance_matrix(x=df.x, y=df.y, names=df.parnme)
            pyemu.helpers.first_order_pearson_tikhonov(pst, cov, reset=False, abs_drop_tol=0.2)

    def _default_pest_args(self, pst):
        pst.control_data.noptmax = self.base.get("noptmax")
        pst.control_data.phiredstp = 1.000000e-002
        pst.control_data.nphistp = 3
        pst.control_data.nphinored = 3
        pst.control_data.relparstp = 1.000000e-002
        pst.control_data.nrelpar = 3
        pst.reg_data.phimlim = self.base.get("phimlim")
        pst.reg_data.phimaccept = pst.reg_data.phimlim * 1.1
        pst.svd_data.svdmode = 1
        pst.svd_data.maxsing = 604
        pst.svd_data.eigthresh = 5.000000e-007
        pst.control_data.rlambda1 = 2.000000e+001
        pst.control_data.rlamfac = -3.000000e+000
        pst.control_data.phiratsuf = 3.000000e-001
        pst.control_data.phiredlam = 1.000000e-002
        pst.control_data.numlam = 7
        pst.control_data.jacupdate = 999
        pst.control_data.lamforgive = "lamforgive"
        pst.write(os.path.join(self.tmp_model_ws, self.pestname))

    def execute_pest(self):
        self._add_observations()
        pp_names = self._add_parameters()
        self._prepare_run()
        pst = self.create_pst()
        if self.base.get("first_reg"):
            self._add_first_regularization(pst, pp_names)
        if self.base.get("zero_reg"):
            pyemu.helpers.zero_order_tikhonov(pst, reset=True)
        self._default_pest_args(pst)
        if not self.onlycreate:
            num_workers = 8
            pyemu.utils.os_utils.start_workers(self.tmp_model_ws, self.pest_version, self.pestname,
                                               num_workers=num_workers, worker_root='.',
                                               verbose=True, master_dir=self.clb_model_ws)
        else:
            return pst

    def apply_pest(self):
        self._move_model(self.clb_model_ws, f"{self.clb_model_ws}_temp")
        pst = pyemu.Pst(os.path.join(self.clb_model_ws, self.pestname))
        pst.parrep(os.path.join(self.clb_model_ws, f"{Path(self.pestname).stem}.par"))
        pst.write_input_files(pst_path=self.clb_model_ws)
        pst.write(os.path.join(self.clb_model_ws, self.pestname))
        pyemu.os_utils.run(f"{self.pest_version} {self.pestname}", cwd=self.clb_model_ws)
        self.sim, self.model = self._get_model(self.clb_model_ws)
