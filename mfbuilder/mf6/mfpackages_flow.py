import numpy as np
from flopy.mf6 import ModflowGwfnpf, ModflowGwfrcha,ModflowGwfrch, ModflowGwfic
from flopy.mf6.modflow import ModflowUtltvk
from mfbuilder.dto.base import ProjectConfig


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
        idomain_lay1 = self.grid.idomain[0]
        target_indices = np.where(idomain_lay1 == 1)[0]
        spd_layer1 = []
        for cell_idx in target_indices:
            spd_layer1.append([(0, cell_idx), 'k', 1.0])
            spd_layer1.append([(0, cell_idx), 'k22', 1.0])
            spd_layer1.append([(0, cell_idx), 'k33', 0.1])

        tvk_perioddata = {
            1: spd_layer1
        }
        npf = ModflowGwfnpf(
            self.model,
            icelltype=self.cfg.npf.icelltype,
            k=hk,
            k22=k22,
            k33=k33,
            angle1=anglx,
            angle2=angly,
            angle3=anglz,
        )
        tvk = ModflowUtltvk(
            npf,
            perioddata=tvk_perioddata
        )


    def _build_ic(self):
        strt = self.cfg.ic.load_array(self.grid)
        ModflowGwfic(self.model, strt=strt)

    def _build_rch(self):
        rch_cfg = self.cfg.rch

        if isinstance(rch_cfg, dict):
            rch_spd = {int(per): cfg.load_array(self.grid) for per, cfg in rch_cfg.items()}
        else:
            rch_spd = {0: rch_cfg.load_array(self.grid)}
        irch_array = np.ones(self.grid.ncpl, dtype=int)
        idomain_lay1 = self.grid.idomain[0]
        idx_inactive = (idomain_lay1 < 1)
        irch_array2 = np.zeros(self.grid.ncpl, dtype=int)
        irch_array2[idx_inactive] = 1

        irch = {0: irch_array, 1: irch_array2}

        ModflowGwfrcha(self.model, readasarrays=True, recharge=rch_spd, irch=irch)
        # ModflowGwfrcha(self.model, readasarrays=True, recharge=rch_spd)
        # ModflowGwfrch(self.model, stress_period_data=rch_spd)
