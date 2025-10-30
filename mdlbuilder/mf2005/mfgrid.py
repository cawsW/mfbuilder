from flopy.modflow import Modflow, ModflowDis

from mdlbuilder.mdlbase.mfgrid import StructuredGridBuilder


class StructuredGridMf2005Builder(StructuredGridBuilder):
    def props_grid(self):
        modelgrid = self._create_temp_dis()
        ncpl = (modelgrid.nrow, modelgrid.ncol)
        top, botm = self._process_surface(ncpl, modelgrid)
        return dict(
            nlay=self.data.nlay,
            nrow=self.data.nrow,
            ncol=self.data.ncol,
            delr=self.data.cell_size,
            delc=self.data.cell_size,
            top=top,
            botm=botm,
            xmin=self.data.xmin,
            ymax=self.data.ymax,
            ymin=self.data.ymin,
            crs=self.data.epsg,
        )

    def create_grid(self, model: Modflow):
        ModflowDis(model, **self.props_grid())
        return model.modelgrid