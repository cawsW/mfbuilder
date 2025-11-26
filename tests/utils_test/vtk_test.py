import stat
import os
from mfbuilder.workingTools import Mf6VtkGenerator
exeName = "../bin/mf6"
st = os.stat(exeName)
os.chmod(exeName, st.st_mode | stat.S_IEXEC)

vtkGen = Mf6VtkGenerator(simName="mfsim.nam",
                         simWs="raspad",
                         exeName=exeName,
                         vtkDir='vtk_out')
vtkGen.loadSimulation()
vtkGen.loadModel('rsp')
vtkGen.generateGeometryArrays()
vtkGen.generateParamVtk()
vtkGen.generateBcObsVtk()