from flopy.mf6 import (ModflowGwf, ModflowGwfriv, ModflowGwfdrn,
                       ModflowGwfghb, ModflowGwfwel)
from mfbuilder.mdlbase.mixins import VertexGridMixin, StructuredGridMixin

class MF6StructuredRivHandler(StructuredGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfriv(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None):
        icell = tuple(icell)
        vals = cache.resolve_all(icell)
        return [(layer - 1, *icell), vals["stage"], vals["cond"], vals["elev"], bname]


class MF6VertexRivHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfriv(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None):
        vals = cache.resolve_all(icell)
        return [(layer - 1, icell), vals["stage"], vals["cond"], vals["elev"], bname]


class MF6VertexGhbHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfghb(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None):
        vals = cache.resolve_all(icell)
        return [(layer - 1, icell), vals["bhead"], vals["cond"], bname]


class MF6VertexDrnHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfdrn(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None):
        vals = cache.resolve_all(icell)
        return [(layer - 1, icell), vals["head"], vals["cond"], bname]


class MF6VertexWelHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfwel(model,  stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None):
        vals = cache.resolve_all(icell)
        return [(layer - 1, icell), vals["rate"], bname]
