import os
import shutil
import stat
import numpy as np
import flopy
import pandas as pd
import pyemu


ws = 'raspad'
model_name = 'rsp'

tmp_model_ws = os.path.join('.', 'pest_copy')
template_ws = os.path.join('.', 'pest_reg')

exe_name = os.path.join("..", "bin", "mf6")
st = os.stat(exe_name)
os.chmod(exe_name, st.st_mode | stat.S_IEXEC)

sim = flopy.mf6.MFSimulation.load(sim_ws=ws, exe_name=exe_name)
m = sim.get_model(model_name)
modelgrid = m.modelgrid


columns_river = ['lay', 'icpl', 'stage', 'cond', 'botm', 'zone']
river_zones = pd.read_csv(os.path.join(ws, f"{model_name}.riv_stress_period_data_1.txt"),
                          sep="\s+", header=None, names=columns_river)
river_zones = river_zones[river_zones['lay'] == 1]
river_zones["zone"] = river_zones["zone"].str.replace('zn_', '').astype(int)
river_zones["zone"] = river_zones["zone"] + 1
cells = pd.Series([cell[0] + 1 for cell in modelgrid.cell2d])
df_s = cells.to_frame(name='icpl')
merged_df = pd.merge(df_s, river_zones, on='icpl', how='left')
merged_df['zone'].fillna(0, inplace=True)
merged_df.rename(columns={'zone': 'zone_ex'}, inplace=True)
# zn_arr1 = np.array([merged_df.zone_ex.to_numpy()])
# zn_arr = np.tile(zn_arr1, 3)[0]
zn_arr1 = merged_df.zone_ex.tolist()
zn_arr =np.array([zn_arr1, zn_arr1])
# sr = modelgrid.cell2d
# sr = {(0, cell[0]): [cell[1], cell[2]] for cell in sr}
# sim.run_simulation()

if os.path.exists(tmp_model_ws):
    shutil.rmtree(tmp_model_ws)
shutil.copytree(ws, tmp_model_ws)
txtfiles = [
    f'{model_name}.npf_k_layer1.txt',
    f'{model_name}.npf_k_layer2.txt',
    f'{model_name}.npf_k_layer3.txt',
    f'{model_name}.npf_k33_layer1.txt',
    f'{model_name}.npf_k33_layer2.txt',
    f'{model_name}.npf_k33_layer3.txt',
    f'{model_name}.rcha_recharge_1.txt',
]
for filet in txtfiles:
    with open(os.path.join(tmp_model_ws, filet), 'r') as file:
        # read all lines
        lines = file.readlines()

    # initialize an empty list
    numbers = []

    # iterate over the lines
    for line in lines:
        # split the line into words
        words = line.split()
        # iterate over the words
        for word in words:
            number = float(word)
            # if successful, append the number to the list
            numbers.append(number)
    np.savetxt(os.path.join(tmp_model_ws, filet), np.array(numbers))

pf = pyemu.utils.PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                         remove_existing=True, longnames=True, spatial_reference=modelgrid, zero_based=False)

hds_df = pf.add_observations(f"{model_name}.obs.head.csv", insfile=f"{model_name}.obs.head.csv.ins", index_cols="time",
                             prefix="hds", obsgp="heads1")

vchd = pyemu.geostats.ExpVario(a=3500, contribution=1)
grid_chd = pyemu.geostats.GeoStruct(variograms=vchd, transform="log")

vchd2 = pyemu.geostats.ExpVario(a=2000, contribution=1)
grid_chd2 = pyemu.geostats.GeoStruct(variograms=vchd2, transform="log")

pf.add_parameters(filenames=f"{model_name}.riv_stress_period_data_1.txt", par_type="zone",
                  zone_array=zn_arr,
                  par_name_base="riv_layer_1", pargp="riv_layer_1", index_cols=[0, 1], use_cols=[3],
                  upper_bound=100, lower_bound=0.0001, initial_value=1, ult_ubound=1000, ult_lbound=0.0001)

# for lay in [0, 2]:
#     pf.add_parameters(filenames=f"{model_name}.npf_k_layer{lay + 1}.txt", par_type="constant",
#                       par_name_base=f"hk_layer_{lay + 1}", pargp=f"hk_layer_{lay + 1}",
#                       upper_bound=20, lower_bound=0.001,  initial_value=1, ult_ubound=70, ult_lbound=0.0001)
#
# for lay in [0, 2]:
#     pf.add_parameters(filenames=f"{model_name}.npf_k33_layer{lay + 1}.txt", par_type="constant",
#                       par_name_base=f"k33_layer_{lay + 1}", pargp=f"k33_layer_{lay + 1}",
#                       upper_bound=100, lower_bound=0.001,  initial_value=1, ult_ubound=2, ult_lbound=0.02)
#
# pf.add_parameters(filenames=f"{model_name}.npf_k33_layer2.txt", par_type="pilotpoint",
#                   par_name_base="k33_layer_2", pargp="k33_layer_2",
#                   pp_space="pp.shp", geostruct=grid_chd,
#                   upper_bound=100, lower_bound=0.001,  initial_value=1, ult_ubound=2, ult_lbound=0.02)
#
# pf.add_parameters(filenames=f"{model_name}.npf_k_layer2.txt", par_type="pilotpoint",
#                   par_name_base="hk_layer_2", pargp="hk_layer_2",
#                   pp_space="pp.shp", geostruct=grid_chd,
#                   upper_bound=20, lower_bound=0.001,  initial_value=1, ult_ubound=70, ult_lbound=0.0001)

pf.add_parameters(filenames=f"{model_name}.rcha_recharge_1.txt", par_type="constant",
                  par_name_base="rch_layer_1", pargp="rch_layer_1",
                  upper_bound=3.5, lower_bound=0.5,  initial_value=1, ult_ubound=0.0007, ult_lbound=0.00001)


pst_name = 'case1_reg.pst'
pest_exe = 'pestpp-glm'
pf.mod_sys_cmds.append(f"mf6 {model_name}.nam")
pst = pf.build_pst(pst_name)
print(pst.observation_data)
cov = pf.build_prior()
pst.control_data.noptmax = 0
# pst.parrep(os.path.join(tmp_model_ws, "case1_reg.par"))
pst.write(os.path.join(template_ws, pst_name))
pyemu.os_utils.run(f"{pest_exe} {pst_name}", cwd=template_ws)
# pst.write_input_files(pst_path=tmp_model_ws)
# pp_name = {f"hk_layer_{lay + 1}_inst0pp.dat.tpl": grid_chd for lay in [1]}
# pp_name.update({f"k33_layer_{lay + 1}_inst0pp.dat.tpl": grid_chd for lay in [1]})
# pp_name.update({"rch_layer_1_inst0pp.dat.tpl": grid_chd2})
# for k, v in pp_name.items():
#     df = pyemu.pp_utils.pp_tpl_to_dataframe(os.path.join(template_ws, k))
#     cov = v.covariance_matrix(x=df.x, y=df.y, names=df.parnme)
#     pyemu.helpers.first_order_pearson_tikhonov(pst, cov, reset=False, abs_drop_tol=0.2)

pst_name = 'case1_reg.pst'
# pyemu.helpers.zero_order_tikhonov(pst, parbounds=False, par_groups=["riv_layer_1"], reset=True)
# pst.adjust_weights_discrepancy(bygroups=True)
pst.control_data.noptmax = 25
pst.control_data.phiredstp = 1.000000e-002
pst.control_data.nphistp = 3
pst.control_data.nphinored = 3
pst.control_data.relparstp = 1.000000e-002
pst.control_data.nrelpar = 3
pst.reg_data.phimlim = 2900
pst.reg_data.phimaccept = 3190
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