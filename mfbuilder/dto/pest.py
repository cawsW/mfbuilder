"""Конфигурация сборки PEST(++)-контроля через pyemu.PstFrom.

Аналог главного конфига модели (mfbuilder.dto.base.ProjectConfig), но для
калибровки: вместо сборки MODFLOW-пакетов из растров/векторов — сборка
.pst/.tpl/.ins из уже посчитанной модели (нужен готовый model_ws с внешними
текстовыми файлами массивов, т.е. builder.finalize() с write_input=True).
"""
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# --- Геостатистика (пилотные точки) -----------------------------------------

class VariogramConfig(BaseModel):
    type: Literal["spherical", "exponential", "gaussian"] = "spherical"
    range: float = Field(..., gt=0, description="Радиус корреляции (a), м")
    contribution: float = Field(default=1.0, gt=0)
    anisotropy: float = 1.0
    bearing: float = 0.0


class GeoStructConfig(BaseModel):
    variogram: VariogramConfig
    nugget: float = 0.0
    transform: Literal["log", "none"] = "log"


# --- Параметры ---------------------------------------------------------------

class BaseParameterGroupConfig(BaseModel):
    """Общие поля для любого типа параметризации PstFrom.add_parameters."""
    model_config = ConfigDict(extra="forbid")

    files: list[str] = Field(description="Файлы модели (относительно workspace.calib_ws), которые параметризуются")
    par_name_base: str = Field(description="Базовое имя параметра/группы параметров")
    pargp: str | None = Field(default=None, description="Группа параметров (по умолчанию = par_name_base)")

    index_cols: list[int | str] | None = Field(
        default=None, description="Для списочных файлов (stress_period_data) — колонки-идентификаторы строки")
    use_cols: list[int | str] | None = Field(default=None, description="Какие колонки считать параметром")
    use_rows: list | None = Field(default=None, description="Ограничить параметризацию подмножеством строк")

    initial_value: float = 1.0
    lower_bound: float
    upper_bound: float
    ult_lbound: float | None = Field(default=None, description="Абсолютный нижний предел итогового (после мульт.) значения")
    ult_ubound: float | None = Field(default=None, description="Абсолютный верхний предел итогового значения")
    transform: Literal["log", "none"] = "log"

    extra: dict = Field(default_factory=dict, description="Доп. именованные аргументы напрямую в PstFrom.add_parameters")


class ConstantParameterGroup(BaseParameterGroupConfig):
    """Один общий множитель на все files (целиком или на выбранные use_rows/use_cols)."""
    par_type: Literal["constant"] = "constant"


class ZoneSourceConfig(BaseModel):
    """Источник зон для ZoneParameterGroup."""
    file: Path | None = Field(default=None, description="Векторный файл с полем-номером зоны; не задан — зоны из unique-значений самого массива")
    field: str = Field(default="zone")
    default: float = 0.0


class ZoneParameterGroup(BaseParameterGroupConfig):
    """Свой множитель на каждую зону (одна зона = одна ячейка zone_array, 0 исключает ячейку)."""
    par_type: Literal["zone"] = "zone"
    zones: ZoneSourceConfig


class PilotPointParameterGroup(BaseParameterGroupConfig):
    """Пилотные точки + geostruct. Точки — уже готовый файл (name/x/y), генерация не выполняется."""
    par_type: Literal["pilotpoints"] = "pilotpoints"
    points: Path = Field(description="Файл с пилотными точками (shp с полями name/x/y, либо geometry)")
    geostruct: GeoStructConfig
    zones: ZoneSourceConfig | None = Field(default=None, description="Опционально сузить точки/ячейки одной зоной")


class DirectParameterGroup(BaseParameterGroupConfig):
    """par_style='direct' — PEST пишет физическое значение прямо в файл (не множитель).
    Обычно в паре с pre-command хуком, разворачивающим один коэффициент в N файлов модели
    (см. PestConfig.hooks)."""
    par_type: Literal["direct"] = "direct"


ParameterGroupConfig = Annotated[
    Union[ConstantParameterGroup, ZoneParameterGroup, PilotPointParameterGroup, DirectParameterGroup],
    Field(discriminator="par_type"),
]


class TiedParameterConfig(BaseModel):
    """Постфактум pst.parameter_data.partrans='tied' для part_names -> tied_to."""
    par_names: list[str]
    tied_to: str


# --- Наблюдения ----------------------------------------------------------------

class ObservationGroupConfig(BaseModel):
    """Один источник наблюдений (CSV с результатами модели/zonebudget/OBS-пакета).

    Мягкое ограничение-неравенство задаётся тем же классом — префикс obsgp
    'greater_than_'/'less_than_' распознаётся самим PstFrom.
    """
    model_config = ConfigDict(extra="forbid")

    file: str = Field(description="CSV-файл (относительно workspace.calib_ws)")
    obsgp: str
    index_cols: list[str | int]
    use_cols: list[str | int] | None = None
    use_rows: list | None = None
    prefix: str = ""
    weight: float = Field(default=1.0, description="Базовый вес группы (уточняется правилами WeightRule)")
    extra: dict = Field(default_factory=dict, description="Доп. именованные аргументы напрямую в PstFrom.add_observations")


class WeightRule(BaseModel):
    """Правило переопределения weight/obsval у уже собранных наблюдений.

    Применяются по порядку списка PestConfig.weights — каждое следующее
    правило может перекрыть эффект предыдущего на пересекающемся наборе строк.
    Пустой фильтр (все поля None) не задаётся — хотя бы один должен быть указан.
    """
    obsgp: str | None = Field(default=None, description="Точное совпадение группы наблюдений")
    name_contains: str | None = Field(default=None, description="Подстрока в имени наблюдения")
    names: list[str] | None = Field(default=None, description="Явный список имён наблюдений")
    weight: float | None = None
    obsval: float | None = None


# --- Геостатистическая регуляризация (Tikhonov, только GLM) --------------------

class RegularizationConfig(BaseModel):
    zero_order: bool = Field(default=True, description="pyemu.helpers.zero_order_tikhonov (preferred value)")
    pilot_point_groups: list[str] = Field(
        default_factory=list,
        description="pargp пилотных точек, для которых также строится first_order_pearson_tikhonov "
                    "(preferred difference по geostruct этой группы)"
    )
    abs_drop_tol: float = 0.2
    phimlim: float = Field(description="Целевой Phi для регуляризации (reg_data.phimlim)")
    phimaccept: float | None = Field(default=None, description="По умолчанию phimlim * 1.1")


# --- Настройки запуска PEST++ ---------------------------------------------------

class ConvergenceConfig(BaseModel):
    """'Стандартный квинтет' критериев сходимости control_data."""
    phiredstp: float = 1e-2
    nphistp: int = 3
    nphinored: int = 3
    relparstp: float = 1e-2
    nrelpar: int = 3


class SvdConfig(BaseModel):
    svdmode: int = 1
    eigthresh: float = 1e-6
    maxsing: int | None = None


class GlmRunConfig(BaseModel):
    kind: Literal["glm"] = "glm"
    noptmax: int = 7
    rlambda1: float = 20.0
    rlamfac: float = -3.0
    phiratsuf: float = 0.3
    phiredlam: float = 0.01
    numlam: int = 7
    jacupdate: int = 1
    lamforgive: bool = True


class IesRunConfig(BaseModel):
    kind: Literal["ies"] = "ies"
    noptmax: int = 6
    num_reals: int = 100
    initial_lambda: float = 10.0
    lambda_mults: list[float] = Field(default_factory=lambda: [0.1, 1.0, 10.0])
    subset_size: int | None = None
    bad_phi_sigma: float | None = None
    autoadaloc: bool = False
    save_binary: bool = False
    forecasts: list[str] = Field(default_factory=list)
    parameter_ensemble: str | None = Field(default=None, description="Заданный prior (например, для continue/restart)")
    restart_obs_ensemble: str | None = None
    observation_ensemble: str | None = None


class SweepRunConfig(BaseModel):
    kind: Literal["sweep"] = "sweep"
    parameter_csv_file: str
    output_csv_file: str = "sweep_out.csv"


RunConfig = Annotated[Union[GlmRunConfig, IesRunConfig, SweepRunConfig], Field(discriminator="kind")]


# --- Прочее -----------------------------------------------------------------

class PyFunctionHookConfig(BaseModel):
    """Врезка пользовательской функции в forward_run.py (PstFrom.add_py_function)."""
    path: Path
    call: str = Field(description='Вызов, например "collect_heads()"')
    when: Literal["pre", "post"] = "pre"


class WorkersConfig(BaseModel):
    """Параметры pyemu.utils.os_utils.start_workers — заполняется только если нужно запускать отсюда же."""
    exe_name: str
    num_workers: int = 4
    worker_root: Path | None = None
    master_dir: Path | None = None


class PestWorkspaceConfig(BaseModel):
    """Рабочие директории — то же разделение original_d/new_d, что во всех
    существующих pest_*.py скриптах: model_ws не трогается, calib_ws — его
    одноразовая копия (в неё же кладутся служебные файлы зон/точек/CSV),
    template_ws — то, что строит PstFrom (new_d)."""
    model_ws: Path = Field(description="Уже посчитанная модель (base.workspace главного конфига)")
    calib_ws: Path = Field(description="Рабочая копия model_ws — original_d для PstFrom")
    template_ws: Path = Field(description="Шаблон PstFrom — new_d, куда пишутся .tpl/.ins/.pst")
    pst_name: str = "pest.pst"
    exe_paths: list[Path] = Field(default_factory=list, description="Исполняемые файлы, копируемые в calib_ws (mf6, pestpp-*, zbud6)")
    flatten_arrays: bool = Field(default=True, description="Развернуть многозначные строки текстовых массивов в один столбец (нужно pyemu)")


class PestConfig(BaseModel):
    """Корневой конфиг сборки PEST(++)-контроля."""
    workspace: PestWorkspaceConfig
    parameters: list[ParameterGroupConfig]
    observations: list[ObservationGroupConfig] = Field(default_factory=list)
    weights: list[WeightRule] = Field(default_factory=list)
    tied: list[TiedParameterConfig] = Field(default_factory=list)
    run: RunConfig
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    svd: SvdConfig = Field(default_factory=SvdConfig)
    regularization: RegularizationConfig | None = None
    forward_run_commands: list[str] = Field(default_factory=lambda: ["mf6"])
    hooks: list[PyFunctionHookConfig] = Field(default_factory=list)
    workers: WorkersConfig | None = None
