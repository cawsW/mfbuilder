from flopy.discretization import VertexGrid
from flopy.mf6 import ModflowGwfdisv, ModflowGwfdis, ModflowGwf

from mdlbuilder.mdlbase.mfgrid import StructuredGridBuilder, VertexGridBuilder, UnstructuredGridBuilder


class StructuredGridMf6Builder(StructuredGridBuilder):
    def props_grid(self):
        modelgrid = self._create_temp_dis()
        ncpl = (modelgrid.nrow, modelgrid.ncol)
        top, botm = self._process_surface(ncpl, modelgrid)
        idomain = self._active_domain(modelgrid)
        return dict(
            nlay=self.data.nlay,
            nrow=self.data.nrow,
            ncol=self.data.ncol,
            delr=self.data.cell_size,
            delc=self.data.cell_size,
            top=top,
            botm=botm,
            idomain=idomain,
            xorigin=self.data.xmin,
            yorigin=self.data.ymin
        )

    def create_grid(self, model: ModflowGwf):
        ModflowGwfdis(model, **self.props_grid())
        return model.modelgrid


class VertexGridMf6Builder(VertexGridBuilder):
    def props_grid(self):
        gridprops = self._get_gridprops()
        ncpl = gridprops["ncpl"]
        modelgrid = VertexGrid(
            nlay=self.data.nlay,
            ncpl=ncpl,
            vertices=gridprops["vertices"],
            cell2d=gridprops["cell2d"]
        )
        gridprops["top"], gridprops["botm"] = self._process_surface(ncpl, modelgrid)
        return gridprops

    def create_grid(self, model: ModflowGwf):
        ModflowGwfdisv(model, **self.props_grid())
        return model.modelgrid



class UnstructuredGridMf6Builder(UnstructuredGridBuilder):
    def props_grid(self):
        pass

