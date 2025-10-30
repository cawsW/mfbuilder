from typing import Any, Protocol

import numpy as np

from mdlbuilder.dto.types import TModel, TGrid
from mdlbuilder.utils.mfdata import  FieldResolverCache


class IModelBuilder(Protocol):
    def create_sim(self) -> Any: ...
    def finalize(self) -> None: ...
    def run(self) -> None: ...


class IGridBuilder(Protocol):
    def props_grid(self) -> dict[str, Any]: ...
    def create_grid(self, model: TModel) -> TGrid: ...


class ISourceSinksBuilder(Protocol):
    def build_package(self, model: TModel): ...
    def build_record(self, layer: int, icell: int | np.ndarray, cache: FieldResolverCache) -> list: ...


#TODO: протокол для параметров
