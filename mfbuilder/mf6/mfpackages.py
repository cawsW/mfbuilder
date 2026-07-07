from flopy.mf6 import (ModflowGwf, ModflowGwfriv, ModflowGwfdrn,
                       ModflowGwfghb, ModflowGwfwel, ModflowGwflak)
from mfbuilder.mdlbase.mixins import VertexGridMixin, StructuredGridMixin
from mfbuilder.utils.mfdata import FieldResolverCache

class MF6StructuredRivHandler(StructuredGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfriv(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        icell = tuple(icell)
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, *icell), vals["stage"], vals["cond"], vals["elev"], bname]


class MF6VertexRivHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfriv(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, icell), vals["stage"], vals["cond"], vals["elev"], bname]


class MF6VertexGhbHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfghb(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, icell), vals["bhead"], vals["cond"], bname]


class MF6VertexDrnHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfdrn(model, stress_period_data=records, boundnames=True, mover=self.mover)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, icell), vals["head"], vals["cond"], bname]


class MF6VertexWelHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfwel(model,  stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, icell), vals["rate"], bname]


class MF6VertexLakHandler(VertexGridMixin):
    """Озёра/пруды (полигоны) как пакет LAK. Каждый полигон -> одно озеро,
    подключенное вертикально ко всем ячейкам сетки, которые он покрывает."""

    claktype = "VERTICAL"

    def build_package(self, model: ModflowGwf):
        periods = sorted(self.data.keys())
        packagedata = []
        connectiondata = []
        perioddata = {}

        for period in periods:
            zone = self.data[period]
            period_records = []
            lakeno = 0

            for feature in zone.data:
                geom_gdf = feature.get_filtered_geometry()
                resolver_cache = FieldResolverCache(feature, self.grid, geom_gdf)

                for geom_index, geom in enumerate(geom_gdf.geometry):
                    cells = self.map_to_grid(geom)
                    if not cells:
                        continue
                    layers = feature.resolve_layers(geom_gdf, geom_index)
                    vals = resolver_cache.resolve_all(cells[0], geom_index)

                    if period == periods[0]:
                        bname = resolver_cache.resolve_boundname(geom_gdf, geom_index)
                        iconn = 0
                        for layer in layers:
                            for icell in cells:
                                connectiondata.append([
                                    lakeno, iconn, (layer - 1, icell),
                                    self.claktype, vals["cond"], 0.0, 0.0, 0.0, 0.0,
                                ])
                                iconn += 1
                        packagedata.append([lakeno, vals["head"], iconn, bname])

                    period_records.append([lakeno, "evaporation", vals["evaporation"]])
                    period_records.append([lakeno, "runoff", vals["runoff"]])

                    lakeno += 1

            perioddata[int(period)] = period_records

        return ModflowGwflak(
            model,
            boundnames=True,
            nlakes=len(packagedata),
            noutlets=0,
            packagedata=packagedata,
            connectiondata=connectiondata,
            perioddata=perioddata,
        )
