from enum import Enum
from typing import Sequence, TypeVar

Scalar = int | float
ScalarOrPath = Scalar | str
DelArray = Sequence[Scalar]

TModel = TypeVar("TModel")
TGrid = TypeVar("TGrid")

class EngineType(str, Enum):
    MF2005 = "mf2005"
    MF6 = "mf6"


class TimeUnitType(str, Enum):
    YEARS = "YEARS"
    DAYS = "DAYS"
    MINUTES = "MINUTES"
    SECONDS = "SECONDS"


class LengthUnitType(str, Enum):
    FEET = 1
    METERS = 2
    CENTIMETERS = 3


class GridType(str, Enum):
    STRUCTURED = "structured"
    VERTEX = "vertex"
    UNSTRUCTURED = "unstructured"
