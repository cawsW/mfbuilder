from __future__ import annotations

from flopy.mf6 import MFSimulation, ModflowGwf, ModflowIms, ModflowTdis, ModflowGwfoc

from mfbuilder.mfmain import ProjectConfig


def default_ims_kwargs() -> dict:
    """Настройки решателя IMS, общие для сборки модели из конфига (MF6Builder)
    и пересборки из geojson (mfbuilder.export.ModelImporter)."""
    return dict(
        # COMPLEX (не SIMPLE) - нужен из-за нелинейных элементов модели
        # (несколько взаимодействующих WEIR-водосливов в LAK, MVR, транзиентный LAK):
        # на SIMPLE/MODERATE стационарный период не сходится (PACKAGE ...-stage
        # CAUSED CONVERGENCE FAILURE).
        # linear_acceleration=BICGSTAB обязателен при Newton (матрица несимметрична).
        complexity="COMPLEX",
        outer_maximum=500,
        outer_dvclose=1e-4,
        inner_maximum=100,
        inner_dvclose=1e-4,
        under_relaxation="DBD",
        linear_acceleration="BICGSTAB",
    )


class MF6Builder:
    """MODFLOW 6 builders (stub). Later: use flopy.mf6 to make MFSimulation/ModflowGwf etc."""

    def __init__(self, ctx: ProjectConfig) -> None:
        self.ctx = ctx
        self.sim: MFSimulation | None = None
        self.model: ModflowGwf | None = None

    def create_tdis(self) -> None:
        tdis_cfg = self.ctx.tdis
        ModflowTdis(
            self.sim,
            nper=tdis_cfg.nper,
            time_units=self.ctx.base.tunits,
            perioddata=tdis_cfg.perioddata,
        )

    def create_ims(self) -> None:
        ModflowIms(self.sim, **default_ims_kwargs())

    def create_sim(self) -> ModflowGwf:
        cfg = self.ctx.base
        self.sim = MFSimulation(
            sim_name=cfg.name,
            version="mf6",
            exe_name=self.ctx.base.exe_path,
            sim_ws=str(self.ctx.base.workspace),
        )
        self.create_tdis()
        self.create_ims()
        # NEWTON UNDER_RELAXATION — Newton-Raphson с псевдо-транзиентным продолжением.
        # Устраняет DRY/WET-переключения ячеек (особенно в стационарных периодах).
        # UNDER_RELAXATION помогает сходимости при плохих начальных условиях.
        self.model = ModflowGwf(
            self.sim,
            modelname=cfg.name,
            save_flows=True,
            # newtonoptions="UNDER_RELAXATION",
            newtonoptions="NEWTON",
            # newtonoptions="NEWTON UNDER_RELAXATION",
        )
        return self.model

    def finalize(self) -> None:
        ModflowGwfoc(
            self.model,
            pname="oc",
            budget_filerecord=f"{self.ctx.base.name}.cbb",
            budgetcsv_filerecord=f"{self.ctx.base.name}.cbb.csv",
            head_filerecord=f"{self.ctx.base.name}.hds",
            headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
            printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )
        self.sim.set_all_data_external(True)
        self.sim.write_simulation()

    def run(self) -> None:
        self.sim.run_simulation()