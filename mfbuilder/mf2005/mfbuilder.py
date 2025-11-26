from __future__ import annotations

from flopy.modflow import Modflow

from mfbuilder.mfmain import ProjectConfig


class MF2005Builder:
    """MODFLOW-2005 builders (stub). Later: use flopy.modflow.Modflow and DIS/LPF/etc."""

    def __init__(self, ctx: ProjectConfig) -> None:
        self.ctx = ctx
        self.model: Modflow | None = None

    def create_sim(self):
        cfg = self.ctx.base
        self.model = Modflow(modelname=cfg.name, model_ws=str(self.ctx.base.workspace),
                                       exe_name=cfg.exe_path)
        return self.model

    def finalize(self) -> None:
        (self.ctx.base.workspace / "_mf2005_inputs_stub.txt").write_text("inputs would be written here")

    def run(self) -> None:
        # Deferred: success, output handling
        pass



