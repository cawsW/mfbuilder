from typing import Generic

from mdlbuilder.dto.types import TGrid
from mdlbuilder.dto.packages import SourceSinksZone
from mdlbuilder.utils.mfdata import  FieldResolverCache


class BaseSourceSinksHandler(Generic[TGrid]):
    """Базовый обработчик пакета источников/стоков."""

    def __init__(self, grid: TGrid, data: dict[int, SourceSinksZone]):
        """
        grid: объект сетки (StructuredGrid / VertexGrid)
        data: список SourceFeature (уже провалидированных)
        """
        self.grid = grid
        self.data = data

    def map_to_grid(self, geom):
        gdf = self.grid.geo_dataframe
        mask = gdf.intersects(geom)
        return gdf.index[mask].to_list()

    def iterate_features(self, build_record_fn):
        """Единая логика обхода feature → geometry → cells → record."""
        records = {}

        for spd, feature in self.data.items():
            cellmap = {}

            for f in feature.data:
                geom_gdf = f.load_geometry(f.geometry)
                resolver_cache = FieldResolverCache(f, self.grid, geom_gdf)

                for geom_index, geom in enumerate(geom_gdf.geometry):
                    cells = self.map_to_grid(geom)
                    layers = f.resolve_layers(geom_gdf, geom_index)

                    for layer in layers:
                        for icell in cells:
                            record = build_record_fn(layer, icell, resolver_cache)
                            key = (layer, icell)
                            cellmap[key] = record  # перезапишет, если уже был

            records[spd] = list(cellmap.values())
        return records
