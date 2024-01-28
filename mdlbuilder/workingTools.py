import os, re, time
import flopy
import sys
import numpy as np
import pyvista as pv

class Mf6VtkGenerator:
    def __init__(self, simName, simWs, exeName, vtkDir):
        self.simName = simName
        self.simWs = simWs
        self.exeName = exeName
        self.vtkDir = vtkDir
    
    def loadSimulation(self):
        self.deleteAllFiles()
        sim = flopy.mf6.MFSimulation.load(self.simName, 
                                          exe_name=self.exeName, 
                                          sim_ws=self.simWs)
        self.sim = sim
        print("\n Models in simulation: %s"%self.sim.model_names)
        
    def loadModel(self, modelName):
        self.gwf = self.sim.get_model(modelName)
        self.packageList = self.gwf.get_package_list()
        print("Package list: %s"%self.packageList)
    
    #utils
    def deleteAllFiles(self):
        try:
            # Check if the folder exists
            if not os.path.exists(self.vtkDir):
                os.mkdir(self.vtkDir)
                print(f"Folder created: {file_path}")

            # Iterate through all files in the folder and delete them
            print(f"Check for files in '{self.vtkDir}'.")
            for filename in os.listdir(self.vtkDir):
                file_path = os.path.join(self.vtkDir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")

            print(f"\nAll files in '{self.vtkDir}' have been deleted.\n")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    def saveArray(self,outputDir, array, arrayName):
        if array is not None:
            np.save(os.path.join(outputDir,'param_'+arrayName),array)    
            
    def timing_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Vtk file took {execution_time:.4f} seconds to be generated.")
            return result
        return wrapper
    
    @timing_decorator
    def exportBc(self,bcon):
        #open package and filter names
        bcObj = self.gwf.get_package(bcon)  
        bcObjSpdNames = bcObj.stress_period_data.dtype.names[1:-2]
        #create a flat index
        flatIndexList = []
        for row in bcObj.stress_period_data.data[0]:
            flatIndex = np.ravel_multi_index(row.cellid,self.gwf.modelgrid.shape)
            flatIndexList.append(flatIndex)
        #empty object
        
        #loop over the cells
        if not re.search('rch',bcon,re.IGNORECASE):
            cropVtk = pv.UnstructuredGrid()
            for index, cell in enumerate(flatIndexList):
                tempVtk = self.vtkGeom.extract_cells(cell)
                for name in bcObjSpdNames:
                    tempVtk.cell_data[name] = bcObj.stress_period_data.data[0][name][index]
                cropVtk += tempVtk
        else:
            cropVtk = self.vtkGeom.extract_cells(flatIndexList)
            for name in bcObjSpdNames:
                cropVtk.cell_data[name] = bcObj.stress_period_data.data[0][name]
                
        #save and return
        cropVtk.save(os.path.join(self.vtkDir,bcon+'.vtk'))
        #return cropVtk
        
    @timing_decorator
    def exportObs(self,obs):
        #open package 
        obsObj = self.gwf.get_package(obs)
        obsKey = list(obsObj.continuous.data.keys())[0]
        obsObjNames = obsObj.continuous.data[obsKey].dtype.names[:2]
        print(obsObj.continuous.data[obsKey].dtype.names)
        bcObjArray = obsObj.continuous.data[obsKey].id
        #create a flat index
        flatIndexList = []
        for cell in bcObjArray:
            flatIndex = np.ravel_multi_index(cell,self.gwf.modelgrid.shape)
            flatIndexList.append(flatIndex)
        #empty object
        cropVtk = pv.UnstructuredGrid()
        #loop over the cells
        for index, cell in enumerate(flatIndexList):
            tempVtk = self.vtkGeom.extract_cells(cell)
            for name in obsObjNames:
                tempVtk.cell_data[name] = np.array([obsObj.continuous.data[obsKey][name][index]])
            cropVtk += tempVtk
        #save and return
        cropVtk.save('vtk_out/%s.vtk'%obs)
        #return cropVtk
        
    def generateGeometryArrays(self):
        dis = self.gwf.get_package('DIS')
        dis.export(self.vtkDir, fmt='vtk', binary=False)
        
        if self.gwf.get_package('DIS') is not None:
            dis = self.gwf.get_package('DIS')
            idomainArray = dis.idomain.array
            self.saveArray(self.vtkDir, idomainArray, 'idomain')

        if self.gwf.get_package('IC') is not None:
            ic = self.gwf.get_package('IC')
            strtArray = ic.strt.array
            self.saveArray(self.vtkDir, strtArray, 'strt')

        if self.gwf.get_package('STO') is not None:
            sto = self.gwf.get_package('STO')
            iconvertArray = sto.iconvert.array
            syArray = sto.sy.array
            ssArray = sto.ss.array
            self.saveArray(self.vtkDir, iconvertArray, 'iconvert')
            self.saveArray(self.vtkDir, syArray, 'sy')
            self.saveArray(self.vtkDir, ssArray, 'ss')

        if self.gwf.get_package('NPF') is not None:
            npf = self.gwf.get_package('NPF')
            icelltypeArray = npf.icelltype.array
            kArray = npf.k.array
            k22Array = npf.k22.array
            k33Array = npf.k33.array
            wetdryArray = npf.wetdry.array
            self.saveArray(self.vtkDir, icelltypeArray, 'icelltype')
            self.saveArray(self.vtkDir, kArray, 'k')
            self.saveArray(self.vtkDir, k22Array, 'k22')
            self.saveArray(self.vtkDir, k33Array, 'k33')
            self.saveArray(self.vtkDir, wetdryArray, 'wetdry')
        
        # Build geometry vtk
        vtkFile = os.path.join(self.vtkDir,"disv.vtk")
        gwfGrid = pv.read(vtkFile)
        gwfGrid.clear_cell_data()
        # self.geomPath = ('../Vtk/modelGeometry.vtk')
        self.geomPath = os.path.join("vtk_in", "modelGeometry.vtk")
        print(gwfGrid)
        gwfGrid.save(self.geomPath)
        os.remove(vtkFile)
        self.vtkGeom = gwfGrid
        
    def generateParamVtk(self):
        vtkParam = pv.read(self.geomPath)
        paramList = [file for file in os.listdir(self.vtkDir) if file.startswith('param')]
        
        for param in paramList:
            paramName = re.split('[_;.]',param)[1]
            paramValues = np.load(os.path.join(self.vtkDir,param),allow_pickle=True)
            vtkParam.cell_data[paramName] = paramValues.flatten()
        paramPath = os.path.join(self.vtkDir,'modelParameters.vtk')
        vtkParam.save(paramPath)
        print("Parameter Vtk Generated")
        
    def generateBcObsVtk(self):
        vtkGrid = pv.read(self.geomPath)
        bcList = [x for x in self.packageList if re.search(r'\d',x) and not re.search('obs',x,re.IGNORECASE) and not re.search('rch',x,re.IGNORECASE)]
        obsList = [x for x in self.packageList if re.search('obs',x,re.IGNORECASE)]
        for bc in bcList:
            self.exportBc(bc)
            print("%s vtk generated"%bc)
        for obs in obsList:
            self.exportObs(obs)
            print("%s btk generated"%obs)
            
        