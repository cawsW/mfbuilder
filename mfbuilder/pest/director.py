from pathlib import Path

import pyemu

from mfbuilder.dto.pest import PestConfig, PilotPointParameterGroup, IesRunConfig
from mfbuilder.pest.control import RunControlFactory
from mfbuilder.pest.observations import ObservationBuilder
from mfbuilder.pest.parameters import ParameterStrategyFactory, PilotPointParameterStrategy
from mfbuilder.pest.regularization import RegularizationBuilder
from mfbuilder.pest.weights import TiedParameterApplier, WeightRuleEngine
from mfbuilder.pest.workspace import WorkspacePreparer


class PestDirector:
    """Собирает .pst из PestConfig — аналог mfbuilder.mfmain.Director, но для
    калибровки: на вход не растры/геометрия, а уже посчитанная модель (grid +
    workspace с внешними текстовыми файлами массивов, т.е. написанная с
    outputs.write_input=True).

    Сам Director не знает деталей ни одного шага — только вызывает публичный
    интерфейс более мелких builder'ов, каждый с одной ответственностью
    (SOLID: Dependency Inversion + Single Responsibility). Внедрение через
    конструктор — те же builder'ы можно подменить в тестах или расширить
    новым par_type/run kind через их фабрики, не трогая сам Director.
    """

    def __init__(
        self,
        parameter_factory: ParameterStrategyFactory | None = None,
        run_control_factory: RunControlFactory | None = None,
        observation_builder: ObservationBuilder | None = None,
        weight_engine: WeightRuleEngine | None = None,
        tied_applier: TiedParameterApplier | None = None,
    ) -> None:
        self.parameter_factory = parameter_factory or ParameterStrategyFactory()
        self.run_control_factory = run_control_factory or RunControlFactory()
        self.observation_builder = observation_builder or ObservationBuilder()
        self.weight_engine = weight_engine or WeightRuleEngine()
        self.tied_applier = tied_applier or TiedParameterApplier()

    def build(self, cfg: PestConfig, grid) -> pyemu.Pst:
        calib_ws = WorkspacePreparer(cfg.workspace).prepare(cfg.parameters)

        pf = pyemu.utils.PstFrom(
            original_d=str(calib_ws),
            new_d=str(cfg.workspace.template_ws),
            remove_existing=True,
            longnames=False,
            spatial_reference=grid,
            zero_based=False,
        )

        for group in cfg.parameters:
            strategy = self.parameter_factory.get(group.par_type)
            strategy.add(pf, group, grid, calib_ws)

        for obs_group in cfg.observations:
            self.observation_builder.add(pf, obs_group)

        for cmd in cfg.forward_run_commands:
            pf.mod_sys_cmds.append(cmd)
        for hook in cfg.hooks:
            pf.add_py_function(str(hook.path), hook.call, is_pre_cmd=(hook.when == "pre"))

        pst = pf.build_pst(cfg.workspace.pst_name)

        self.weight_engine.apply(pst, cfg.weights)
        self.tied_applier.apply(pst, cfg.tied)

        self.run_control_factory.get(cfg.run.kind).configure(pst, cfg.run, cfg.convergence, cfg.svd)

        if cfg.regularization is not None:
            geostructs = self._pilot_point_geostructs(cfg)
            RegularizationBuilder(geostructs).apply(pst, cfg.regularization)

        if isinstance(cfg.run, IesRunConfig) and cfg.run.parameter_ensemble is None:
            self._draw_prior_ensemble(pf, pst, cfg)

        pst.write(str(Path(cfg.workspace.template_ws) / cfg.workspace.pst_name))

        if cfg.workers is not None:
            self._launch_workers(cfg)

        return pst

    @staticmethod
    def _pilot_point_geostructs(cfg: PestConfig) -> dict[str, pyemu.geostats.GeoStruct]:
        return {
            (group.pargp or group.par_name_base): PilotPointParameterStrategy.geostruct(group.geostruct)
            for group in cfg.parameters
            if isinstance(group, PilotPointParameterGroup)
        }

    @staticmethod
    def _draw_prior_ensemble(pf: pyemu.utils.PstFrom, pst: pyemu.Pst, cfg: PestConfig) -> None:
        pe = pf.draw(num_reals=cfg.run.num_reals)
        pe.to_csv(str(Path(cfg.workspace.template_ws) / "prior_pe.csv"))
        pst.pestpp_options["ies_parameter_ensemble"] = "prior_pe.csv"

    @staticmethod
    def _launch_workers(cfg: PestConfig) -> None:
        workers = cfg.workers
        pyemu.utils.os_utils.start_workers(
            str(cfg.workspace.template_ws),
            workers.exe_name,
            cfg.workspace.pst_name,
            num_workers=workers.num_workers,
            worker_root=str(workers.worker_root) if workers.worker_root else "..",
            master_dir=str(workers.master_dir) if workers.master_dir else str(cfg.workspace.calib_ws),
            verbose=True,
        )
