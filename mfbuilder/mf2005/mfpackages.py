from flopy.modflow import Modflow, ModflowRiv
from mfbuilder.mdlbase.mixins import StructuredGridMixin

class MF2005StructuredRivHandler(StructuredGridMixin):
    def build_package(self, model: Modflow):
        records = self.iterate_features(self.build_record)
        return ModflowRiv(model, stress_period_data=records)

    def build_record(self, layer, icell, cache):
        icell = tuple(icell)
        vals = cache.resolve_all(icell)
        return [layer - 1, *icell, vals["stage"], vals["cond"], vals["elev"]]
