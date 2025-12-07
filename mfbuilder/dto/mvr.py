from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from shapely.geometry import base as shapely_base


class MvrEndpoint(BaseModel):
    pkg: str
    boundname: str | list[str] | None = None


class MvrFeature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    geometry: Path | shapely_base.BaseGeometry | list[shapely_base.BaseGeometry]
    from_: MvrEndpoint = Field(alias="from")
    to: MvrEndpoint
    factor: float = 1.0


class MvrPeriod(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: list[MvrFeature]


class MvrConfig(RootModel[dict[int, MvrPeriod]]):
    """Конфигурация MVR, сгруппированная по stress-периодам."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_periods(cls, value: Any):
        if isinstance(value, dict):
            return {int(k): v for k, v in value.items()}
        return value

    def __iter__(self):
        return iter(self.root.items())
