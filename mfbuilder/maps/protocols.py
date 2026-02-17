from abc import ABC, abstractmethod
from typing import Any, List, Optional
import matplotlib.pyplot as plt
from pydantic import BaseModel

class IMapLayer(ABC):
    def __init__(self, config: BaseModel, global_crs: Optional[str] = None):
        self.config = config
        self.global_crs = global_crs
        self.mappable = None

    @abstractmethod
    def draw(self, ax: plt.Axes) -> None:
        pass

    @abstractmethod
    def get_legend_handles(self) -> List[Any]:
        pass