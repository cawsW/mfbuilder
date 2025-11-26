import numpy as np
from flopy.discretization import VertexGrid, StructuredGrid
from mfbuilder.dto.types import GridType
from mfbuilder.mdlbase.mfpackages import BaseSourceSinksHandler


class StructuredGridMixin(BaseSourceSinksHandler[StructuredGrid]):
    grid_type = GridType.STRUCTURED

    def map_to_grid(self, geom):
        gdf = self.grid.geo_dataframe
        mask = gdf.intersects(geom)
        gdf["i"], gdf["j"] = np.divmod(np.arange(self.grid.nrow * self.grid.ncol), self.grid.ncol)
        return gdf.loc[mask, ['i', 'j']].to_numpy()


class VertexGridMixin(BaseSourceSinksHandler[VertexGrid]):
    grid_type = GridType.VERTEX

    def map_to_grid(self, geom):
        gdf = self.grid.geo_dataframe
        mask = gdf.intersects(geom)
        return gdf.index[mask].to_list()
