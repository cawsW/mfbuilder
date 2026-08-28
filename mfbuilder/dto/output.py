from pydantic import BaseModel, Field

from mfbuilder.dto.zonebudget import ZoneBudgetConfig


class OutputsConfig(BaseModel):
    write_input: bool = True
    run: bool = True
    zonebudget: ZoneBudgetConfig = Field(default_factory=ZoneBudgetConfig)
