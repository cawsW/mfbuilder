from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from mfbuilder.utils.mfdata import RasterHandler


class NpfConfig(BaseModel):
    """Параметры пакета NPF (анизотропия, углы и т.д.)"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: list[float | Path]
    k33overk: bool = Field(default=False, description="Использовать ли k33 как отношение к k")
    k33: list[float | Path] | None = None
    icelltype: list[int] | int = 0

    # новые поля
    k22overk: bool = Field(default=False, description="Использовать ли k22 как отношение к k")
    k22: list[float | Path] | None = None

    angle1: list[float | Path] | None = None
    angle2: list[float | Path] | None = None
    angle3: list[float | Path] | None = None

    def load_arrays(self, grid):
        """Загружает все параметры NPF как 3D numpy-массивы."""

        def _load_list(values):
            """Преобразует список значений или файлов в 3D numpy массив."""
            arrays = []
            for val in values:
                if isinstance(val, (int, float)):
                    arr = np.full(grid.shape[1:], float(val))
                else:
                    arr = RasterHandler(val).resample_to_grid(grid)
                arrays.append(arr)
            return np.array(arrays)

        # --- hk ---
        hk = _load_list(self.k)

        # --- k33 ---
        if self.k33:
            k33_arr = _load_list(self.k33)
            k33 = hk * k33_arr if self.k33overk else k33_arr
        else:
            k33 = hk * 0.1

        # --- k22 ---
        if self.k22:
            k22_arr = _load_list(self.k22)
            k22 = hk * k22_arr if self.k22overk else k22_arr
        else:
            k22 = hk  # по умолчанию изотропия

        # --- углы ---
        def _optional_angle(values):
            if not values:
                return None
            return _load_list(values)

        angle1 = _optional_angle(self.angle1)
        angle2 = _optional_angle(self.angle2)
        angle3 = _optional_angle(self.angle3)

        return hk, k22, k33, angle1, angle2, angle3


class RchConfig(BaseModel):
    rech: float | Path = 0.0

    def load_array(self, grid):
        if isinstance(self.rech, (int, float)):
            return float(self.rech)
        return RasterHandler(self.rech).resample_to_grid(grid)


class IcConfig(BaseModel):
    strt: float | Path = 0.0

    def load_array(self, grid):
        if isinstance(self.strt, (int, float)):
            return np.full(grid.shape[1:], float(self.strt))
        return RasterHandler(self.strt).resample_to_grid(grid)


class FlowPackagesConfig(BaseModel):
    """Группировка всех 'flow' пакетов."""
    npf: NpfConfig | None = None
    rch: dict[int, RchConfig] | RchConfig | None = None
    ic: IcConfig | None = None
