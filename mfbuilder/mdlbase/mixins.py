import numpy as np
from flopy.discretization import VertexGrid, StructuredGrid
from mfbuilder.dto.types import GridType
from mfbuilder.mdlbase.mfpackages import BaseSourceSinksHandler


class StructuredGridMixin(BaseSourceSinksHandler[StructuredGrid]):
    grid_type = GridType.STRUCTURED

    def _prepare_structured_index(self):
        if getattr(self, "_structured_idx_ready", False):
            return
        # Создаём столбцы i/j один раз
        n = self.grid.nrow * self.grid.ncol
        i, j = np.divmod(np.arange(n), self.grid.ncol)
        self.grid_gdf = self.grid_gdf.copy()
        self.grid_gdf["i"] = i
        self.grid_gdf["j"] = j
        self._structured_idx_ready = True

    def map_to_grid(self, geom):
        self._prepare_structured_index()
        gdf = self.grid_gdf
        idx = self._candidate_index(geom)
        subset = gdf if idx is None else gdf.iloc[idx]
        mask = subset.intersects(geom)
        return subset.loc[mask, ['i', 'j']].to_numpy()


class VertexGridMixin(BaseSourceSinksHandler[VertexGrid]):
    grid_type = GridType.VERTEX

    def map_to_grid(self, geom):
        gdf = self.grid_gdf
        idx = self._candidate_index(geom)
        subset = gdf if idx is None else gdf.iloc[idx]
        mask = subset.intersects(geom)
        return subset.index[mask].to_list()
