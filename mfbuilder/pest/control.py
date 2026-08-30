from typing import Protocol

import pyemu

from mfbuilder.dto.pest import (
    RunConfig,
    GlmRunConfig,
    IesRunConfig,
    SweepRunConfig,
    ConvergenceConfig,
    SvdConfig,
)


class IRunConfigurator(Protocol):
    """Одна реализация = одна разновидность запуска PEST++ (glm/ies/sweep).
    SOLID Open/Closed: новый вид запуска добавляется реализацией этого
    протокола и регистрацией в RunControlFactory."""

    def configure(self, pst: pyemu.Pst, run: RunConfig, convergence: ConvergenceConfig, svd: SvdConfig) -> None: ...


def _apply_common(pst: pyemu.Pst, convergence: ConvergenceConfig, svd: SvdConfig) -> None:
    """'Стандартный квинтет' критериев сходимости + svd_data — идентичен во
    всех сценариях запуска, поэтому вынесен из конкретных конфигураторов."""
    cd = pst.control_data
    cd.phiredstp = convergence.phiredstp
    cd.nphistp = convergence.nphistp
    cd.nphinored = convergence.nphinored
    cd.relparstp = convergence.relparstp
    cd.nrelpar = convergence.nrelpar

    pst.svd_data.svdmode = svd.svdmode
    pst.svd_data.eigthresh = svd.eigthresh
    if svd.maxsing is not None:
        pst.svd_data.maxsing = svd.maxsing


class GlmConfigurator:
    def configure(self, pst: pyemu.Pst, run: GlmRunConfig, convergence: ConvergenceConfig, svd: SvdConfig) -> None:
        _apply_common(pst, convergence, svd)
        cd = pst.control_data
        cd.pestmode = "estimation"
        cd.noptmax = run.noptmax
        cd.rlambda1 = run.rlambda1
        cd.rlamfac = run.rlamfac
        cd.phiratsuf = run.phiratsuf
        cd.phiredlam = run.phiredlam
        cd.numlam = run.numlam
        cd.jacupdate = run.jacupdate
        cd.lamforgive = "lamforgive" if run.lamforgive else "nolamforgive"


class IesConfigurator:
    def configure(self, pst: pyemu.Pst, run: IesRunConfig, convergence: ConvergenceConfig, svd: SvdConfig) -> None:
        _apply_common(pst, convergence, svd)
        pst.control_data.pestmode = "estimation"
        pst.control_data.noptmax = run.noptmax

        opts = pst.pestpp_options
        opts["ies_num_reals"] = run.num_reals
        opts["ies_initial_lambda"] = run.initial_lambda
        opts["ies_lambda_mults"] = run.lambda_mults
        opts["ies_autoadaloc"] = run.autoadaloc
        opts["ies_save_binary"] = run.save_binary
        if run.subset_size is not None:
            opts["ies_subset_size"] = run.subset_size
        if run.bad_phi_sigma is not None:
            opts["ies_bad_phi_sigma"] = run.bad_phi_sigma
        if run.forecasts:
            opts["forecasts"] = ",".join(run.forecasts)
        if run.parameter_ensemble:
            opts["ies_parameter_ensemble"] = run.parameter_ensemble
        if run.restart_obs_ensemble:
            opts["ies_restart_obs_en"] = run.restart_obs_ensemble
        if run.observation_ensemble:
            opts["ies_observation_ensemble"] = run.observation_ensemble


class SweepConfigurator:
    def configure(self, pst: pyemu.Pst, run: SweepRunConfig, convergence: ConvergenceConfig, svd: SvdConfig) -> None:
        _apply_common(pst, convergence, svd)
        for key in [k for k in pst.pestpp_options if k.startswith("ies_")]:
            del pst.pestpp_options[key]
        pst.pestpp_options["sweep_parameter_csv_file"] = run.parameter_csv_file
        pst.pestpp_options["sweep_output_csv_file"] = run.output_csv_file


class RunControlFactory:
    """Реестр конфигураторов запуска, ключ — RunConfig.kind."""

    def __init__(self) -> None:
        self._map: dict[str, IRunConfigurator] = {
            "glm": GlmConfigurator(),
            "ies": IesConfigurator(),
            "sweep": SweepConfigurator(),
        }

    def register(self, kind: str, configurator: IRunConfigurator) -> None:
        self._map[kind] = configurator

    def get(self, kind: str) -> IRunConfigurator:
        configurator = self._map.get(kind)
        if configurator is None:
            raise ValueError(f"Неизвестный тип запуска PEST++: {kind}")
        return configurator
