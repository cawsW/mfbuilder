from pathlib import Path
from datetime import date

from typing import Sequence
from typing_extensions import Annotated
try:
    from pydantic import (BaseModel, ConfigDict, Field, ValidationError, model_validator,
                          computed_field, field_validator)
    from pydantic.types import DirectoryPath, FilePath
except Exception as e:
    raise RuntimeError("pydantic is required: pip install pydantic") from e
from mfbuilder.dto.types import EngineType, TimeUnitType, Scalar
from mfbuilder.dto.grid import GridConfig
from mfbuilder.dto.packages import SourcesSinksConfig
from mfbuilder.dto.packages_flow import FlowPackagesConfig
from mfbuilder.dto.observations import ObservationsConfig
from mfbuilder.dto.output import OutputsConfig



class BaseConfig(BaseModel):
    name: Annotated[str, Field(description="Название модели")]
    workspace: Annotated[DirectoryPath, Field(default="../projects/model", description="Директория модели")]
    engine: Annotated[EngineType, Field(default=EngineType.MF6, description="Движок Modflow")]
    steady: Annotated[bool, Field(default=True, description="Стационар/Нестационар")]
    tunits: Annotated[TimeUnitType, Field(default=TimeUnitType.DAYS, description="Единицы измерения времени")]
    exe_path: Annotated[DirectoryPath | FilePath | None, Field(default="../../../bin",
                                                               description="Путь до исполняемого файла Modflow")]

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    @model_validator(mode="after")
    def check_paths(self):
        workspace_path = Path(self.workspace).expanduser().resolve()
        if not workspace_path.exists():
            workspace_path.mkdir(parents=True, exist_ok=True)

        if isinstance(self.exe_path, str):
            self.exe_path = Path(self.exe_path)
        if self.exe_path and not self.exe_path.exists():
            raise ValueError(f"Укажите корректный путь до exe. Директория '{self.exe_path}' не существует.")

        # --- Подстраиваем exe_path под движок ---
        if self.exe_path and self.engine.value not in self.exe_path.name:
            self.exe_path = Path(f"{self.exe_path}/{self.engine.value}")

        return self


class TransientConfig(BaseModel):
    perlen: Annotated[list[Scalar] | Scalar, Field(default=[1], description="Длина stress period")]
    nstp: Annotated[list[int] | int, Field(default=[1], description="Количество шагов на stress period")]
    tsmult: Annotated[list[Scalar] | Scalar, Field(default=[1], description="Множитель шага на временном шаге")]
    start_datetime: Annotated[date | None, Field(default=None, description="Начальная дата и время")]
    steady: Annotated[
        list[bool] | bool, Field(default=[True], description="Список стационарных и нестационарных stress period")]

    @model_validator(mode="after")
    def validate_series_lengths(self):
        lengths = {
            name: len(value)
            for name in ("perlen", "nstp", "tsmult")
            if isinstance((value := getattr(self, name)), list)
        }
        if lengths and len(set(lengths.values())) != 1:
            raise ValueError("perlen, nstp и tsmult должны быть одной длины.")

        return self

    @computed_field
    @property
    def nper(self) -> int:
        return len(self.nstp)

    @computed_field
    @property
    def perioddata(self) -> list[Sequence]:
        return list(zip(self.perlen, self.nstp, self.tsmult))


class ProjectConfig(BaseModel):
    base: BaseConfig
    tdis: TransientConfig = TransientConfig()
    grid: GridConfig
    sources: SourcesSinksConfig
    parameters: FlowPackagesConfig
    observations: ObservationsConfig
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
