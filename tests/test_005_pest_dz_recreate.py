import os
import pyemu

ws = "pest_copy"
pst_name = "case1_reg.pst"
pest_exe = "pestpp-glm"
pst = pyemu.Pst(os.path.join(ws, pst_name))

pst.control_data.noptmax = 0
pst.parrep(os.path.join(ws, "case1_reg.par"))
pst.write(os.path.join(ws, pst_name))
pyemu.os_utils.run(f"{pest_exe} {pst_name}", cwd=ws)