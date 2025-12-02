from typing import Generic

from mfbuilder.dto.types import TGrid
from mfbuilder.dto.packages import SourceSinksZone
from mfbuilder.utils.mfdata import  FieldResolverCache


class BaseSourceSinksHandler(Generic[TGrid]):
    """Базовый обработчик пакета источников/стоков."""

    def __init__(self, grid: TGrid, data: dict[int, SourceSinksZone]):
        """
        grid: объект сетки (StructuredGrid / VertexGrid)
        data: список SourceFeature (уже провалидированных)
        """
        self.grid = grid
        self.data = data
        # Кешируем GeoDataFrame и пространственный индекс, чтобы не строить их на каждом вызове.
        grid_gdf = getattr(self.grid, "geo_dataframe", None)
        if grid_gdf is None:
            raise RuntimeError("Grid не поддерживает geo_dataframe для маппинга.")
        # deep=False — не копируем геометрию, но не трогаем исходный gdf сетки
        self.grid_gdf = grid_gdf.copy(deep=False)
        try:
            self._sindex = self.grid_gdf.sindex
        except Exception:
            self._sindex = None

    def map_to_grid(self, geom):
        gdf = self.grid_gdf
        idx = self._candidate_index(geom)
        subset = gdf if idx is None else gdf.iloc[idx]
        mask = subset.intersects(geom)
        return subset.index[mask].to_list()

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
                    bname = resolver_cache.resolve_boundname(geom_gdf, geom_index)

                    for layer in layers:
                        for icell in cells:
                            record = build_record_fn(layer, icell, resolver_cache, bname)
                            key = (layer, icell)
                            cellmap[key] = record  # перезапишет, если уже был

            records[spd] = list(cellmap.values())
        return records

    def _candidate_index(self, geom):
        """Быстрая фильтрация кандидатов по sindex перед intersects."""
        if self._sindex is None:
            return None
        try:
            return list(self._sindex.query(geom, predicate="intersects"))
        except TypeError:
            # Для rtree без predicate
            return list(self._sindex.query(geom.bounds))
