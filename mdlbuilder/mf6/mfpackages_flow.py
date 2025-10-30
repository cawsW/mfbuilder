from flopy.mf6 import ModflowGwfnpf, ModflowGwfrcha, ModflowGwfic

from mdlbuilder.dto.base import ProjectConfig


class MF6FlowPackageBuilder:
    """Создаёт гидрогеологические пакеты (NPF, RCHA, IC) для MODFLOW 6."""

    def __init__(self, model, grid, cfg: ProjectConfig):
        self.model = model
        self.grid = grid
        self.cfg = cfg.parameters  # FlowPackagesConfig

    def build(self):
        """Основной метод — создаёт пакеты в зависимости от конфигурации."""
        if self.cfg.npf:
            self._build_npf()
        if self.cfg.ic:
            self._build_ic()
        if self.cfg.rch:
            self._build_rch()

    def _build_npf(self):
        hk, k22, k33, anglx, angly, anglz = self.cfg.npf.load_arrays(self.grid)

        ModflowGwfnpf(
            self.model,
            icelltype=self.cfg.npf.icelltype,
            k=hk,
            k22=k22,
            k33=k33,
            angle1=anglx,
            angle2=angly,
            angle3=anglz,
        )

    def _build_ic(self):
        strt = self.cfg.ic.load_array(self.grid)
        ModflowGwfic(self.model, strt=strt)

    def _build_rch(self):
        rech = self.cfg.rch.load_array(self.grid)
        ModflowGwfrcha(self.model, recharge=rech)