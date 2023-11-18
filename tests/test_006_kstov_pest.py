import os
import shutil
import stat
import numpy as np
import flopy
import pyemu

ws = 'kstov'
model_name = 'kstov'

tmp_model_ws = os.path.join('.', 'pest_copy')
template_ws = os.path.join('.', 'pest_reg')

exe_name = os.path.join("..", "bin", "mf6")
st = os.stat(exe_name)
os.chmod(exe_name, st.st_mode | stat.S_IEXEC)

sim = flopy.mf6.MFSimulation.load(sim_ws=ws, exe_name=exe_name)
m = sim.get_model(model_name)
modelgrid = m.modelgrid

if os.path.exists(tmp_model_ws):
    shutil.rmtree(tmp_model_ws)
shutil.copytree(ws, tmp_model_ws)
txtfiles = [
    f'{model_name}.npf_k.txt',
    f'{model_name}.rcha_recharge_1.txt',
]
for filet in txtfiles:
    with open(os.path.join(tmp_model_ws, filet), 'r') as file:
        lines = file.readlines()
    numbers = []
    for line in lines:
        words = line.split()
        for word in words:
            number = float(word)
            numbers.append(number)
    np.savetxt(os.path.join(tmp_model_ws, filet), np.array(numbers).reshape(modelgrid.nrow, modelgrid.ncol))

pf = pyemu.utils.PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                         remove_existing=True, longnames=True, spatial_reference=modelgrid, zero_based=False)

hds_df = pf.add_observations(f"{model_name}.obs.head.csv", insfile=f"{model_name}.obs.head.csv.ins", index_cols="time",
                             prefix="hds", obsgp="heads1")

vchd = pyemu.geostats.ExpVario(a=5000, contribution=0.2)
grid_chd = pyemu.geostats.GeoStruct(variograms=vchd, transform="log")

vchd2 = pyemu.geostats.ExpVario(a=10000, contribution=1)
grid_chd2 = pyemu.geostats.GeoStruct(variograms=vchd2, transform="log")

pf.add_parameters(filenames=f"{model_name}.npf_k.txt", par_type="pilotpoint",
                  par_name_base="hk_layer_1", pargp="hk_layer_1",
                  pp_space="pp.shp", geostruct=grid_chd,
                  upper_bound=20, lower_bound=0.001, initial_value=1, ult_ubound=70, ult_lbound=0.0001)

pf.add_parameters(filenames=f"{model_name}.rcha_recharge_1.txt", par_type="pilotpoint",
                  par_name_base="rch_layer_1", pargp="rch_layer_1",
                  pp_space="pp.shp", geostruct=grid_chd,
                  upper_bound=3.5, lower_bound=0.5, initial_value=1, ult_ubound=0.0007, ult_lbound=0.00001)

# low_weights = ['H_59', 'H_1', 'H_14', 'H_15']
low_weights = ['H_1', 'H_18']
low_weights = '|'.join([f"{w.lower()}_" for w in low_weights])
pst_name = 'case1_reg.pst'
pest_exe = 'pestpp-glm'
pf.mod_sys_cmds.append(f"mf6 {model_name}.nam")
pst = pf.build_pst(pst_name)
pst.observation_data["weight"][pst.observation_data.obsnme.str.contains(low_weights)] = 0.1
cov = pf.build_prior()
pst.control_data.noptmax = 0
pst.write(os.path.join(template_ws, pst_name))
pyemu.os_utils.run(f"{pest_exe} {pst_name}", cwd=template_ws)
pp_name = {"hk_layer_1_inst0pp.dat.tpl": grid_chd, "rch_layer_1_inst0pp.dat.tpl": grid_chd2}
for k, v in pp_name.items():
    df = pyemu.pp_utils.pp_tpl_to_dataframe(os.path.join(template_ws, k))
    cov = v.covariance_matrix(x=df.x, y=df.y, names=df.parnme)
    pyemu.helpers.first_order_pearson_tikhonov(pst, cov, reset=False, abs_drop_tol=0.1)

pst_name = 'case1_reg.pst'
pst.control_data.noptmax = 10
pst.control_data.phiredstp = 1.000000e-002
pst.control_data.nphistp = 3
pst.control_data.nphinored = 3
pst.control_data.relparstp = 1.000000e-002
pst.control_data.nrelpar = 3
# pst.reg_data.phimlim = 550
# pst.reg_data.phimaccept = 610
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
pst.write(os.path.join(template_ws, pst_name))

num_workers = 8
pyemu.utils.os_utils.start_workers(template_ws, pest_exe, pst_name, num_workers=num_workers, worker_root='.',
                                   verbose=True, master_dir=tmp_model_ws)

# send_to_telegram("Pest complete!")