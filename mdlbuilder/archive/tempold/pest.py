import shutil
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import flopy
import pyemu
import itertools
import math

def main():
    ies_exe_path = "pestpp-ies"

    org_model_ws = os.path.join('Narvskaya')
    # sim = flopy.mf6.MFSimulation.load(sim_ws='Narvskaya', verbosity_level=1)
    # sim.continue_=True
    # m = sim.get_model("L_USG")
    # sim.write_simulation()
    # sim.run_simulation()
    os.listdir(org_model_ws)

    tmp_model_ws = "temp_pst_from"
    if os.path.exists(tmp_model_ws):
        shutil.rmtree(tmp_model_ws)
    shutil.copytree(org_model_ws, tmp_model_ws)
    os.listdir(tmp_model_ws)
    nlay = 18
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_model_ws, verbosity_level=0)
    m = sim.get_model("L_USG")

    nodes = os.path.join(org_model_ws, "L_USG.disv_cell2d.txt")
    dt = np.dtype(
        [
            ("node", int),
            ("x", float),
            ("y", float),
            ("num", int),
            ("i1", int),
            ("i2", int),
            ("i3", int),
            ("i4", int),
            ("i5", int),
        ]
    )

    node_ra = np.genfromtxt(nodes, dtype=dt, usecols=(0, 1, 2))  # ,usecols=(0,1,2,4)
    nodelay = os.path.join(org_model_ws, 'gridgen', "qtg.nodesperlay.dat")
    nodelaydat = np.loadtxt(nodelay, dtype=int)
    sr_dict_by_layer = {iter+1: {i[0] - 1: (i[1], i[2]) for i in
                                   node_ra[0:nodelaydat[0]]} for iter in
                        range(len(nodelaydat))}

    sr = {(j, i[0]-1): [i[1], i[2]] for j in range(nlay) for i in node_ra}
    #
    template_ws = "L_USG_template"
    pf = pyemu.utils.PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                             remove_existing=True,
                             longnames=True, spatial_reference=sr,
                             zero_based=False)

    df = pd.read_csv(os.path.join(tmp_model_ws, "L_USG.obs.head.csv"), index_col=0)
    hds_df = pf.add_observations("L_USG.obs.head.csv", insfile="L_USG.obs.head.csv.ins", index_cols="time",
                                 use_cols=list(df.columns.values), prefix="hds", )

    print(hds_df)
    vchd = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
    grid_chd = pyemu.geostats.GeoStruct(variograms=vchd)


    pf.add_parameters(filenames='L_USG.wel_stress_period_data_1.txt', par_type="grid",
                      upper_bound=1.5, lower_bound=0.5,
                      par_name_base="wl", pargp='wells', index_cols=[0, 1], use_cols=[2], geostruct=grid_chd,
                     )

    pf.mod_sys_cmds.append("mf6 L_USG.nam")

    pst = pf.build_pst('L_USG.pst')
    pe = pf.draw(100)  # 100
    pe.to_binary(os.path.join(template_ws, "prior.jcb"))
    pp_par = pst.parameter_data
    changepar = [('well_500', 0, -100, 0, -500), ('well_1000', -100, -500, 0, -1000),
                 ('well_2000', -500, -1000, -100, -2000), ('well_4000', -1000, -2000, -500, -4000),
                 ('well_8000', -2000, -4000, -1000, -8000), ('well_16000', -4000, -8000, -2000, -16000),
                 ('well_32000', -8000, -16000, -4000, -32000)]
    for gr, ub, lb, cub, clb in changepar:
        pp_par.loc[(pp_par.parval1 >= lb) & (pp_par.parval1 < ub), 'pargp'] = gr
        pp_par.loc[pp_par.pargp == gr, 'parlbnd'] = clb
        pp_par.loc[pp_par.pargp == gr, 'parubnd'] = cub
        # pp_par.loc[pp_par.pargp == gr, 'parchglim'] = 'relative'
    # pst.pestpp_options["upgrade_bounds"]=False
    # pp_par.to_csv('par.csv')
    # pst.pestpp_options["ies_par_en"] = "prior.jcb"
    pst.control_data.noptmax = 0
    obs = pst.observation_data

    weig = pd.read_csv(os.path.join(org_model_ws, 'head2.csv'), header=0)
    for i, j in enumerate(weig.well.str.lower()):
        if weig['weight'][i] > 1:
            obs.at[obs.index[obs.loc[:, 'obsnme'].str.contains(j)][0], 'weight'] = 0.1
        else:
            obs.at[obs.index[obs.loc[:, 'obsnme'].str.contains(j)][0], 'weight'] = 0.05
    obs.obgnme = 'head'
    pst.write(os.path.join(template_ws, 'L_USG.pst'))

    pyemu.os_utils.run("{0} L_USG.pst".format(ies_exe_path), cwd=template_ws)

    pst.pestpp_options["ies_par_en"] = "prior.jcb"
    pst.control_data.noptmax = 5
    pst.write(os.path.join(template_ws, 'L_USG.pst'))
    pyemu.os_utils.run("{0} L_USG.pst".format(ies_exe_path), cwd=template_ws)

if __name__ == '__main__':
    main()