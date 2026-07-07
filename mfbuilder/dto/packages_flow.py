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
    strt: dict[int, list[float | Path]] | list[float | Path] = 0.0

    def load_array(self, grid, period: int = 0):
        strt = self.strt
        if isinstance(strt, dict):
            available = sorted(strt.keys())
            # nearest lower period; if none, use the first defined
            selected = next((p for p in reversed(available) if p <= period), available[0])
            strt = strt[selected]
        heads = []
        for ic in strt:
            if isinstance(ic, (int, float)):
                heads.append(np.full(grid.shape[1:], float(ic)))
            else:
                heads.append(RasterHandler(ic).resample_to_grid(grid))
        return heads


class EvtConfig(BaseModel):
    surface: float | Path = 0.0
    rate: float | Path = 0.001
    depth: float | Path = 1.0
    ievt: int | Path | None = None

    def _load_value(self, value, grid, dtype=float):
        if isinstance(value, (int, float)):
            return dtype(value)
        return RasterHandler(value).resample_to_grid(grid)

    def load_arrays(self, grid):
        surface = self._load_value(self.surface, grid, float)
        rate = self._load_value(self.rate, grid, float)
        depth = self._load_value(self.depth, grid, float)
        ievt = None
        if self.ievt is not None:
            ievt = self._load_value(self.ievt, grid, int)
        return surface, rate, depth, ievt

class StoConfig(BaseModel):
    """Параметры пакета STO (хранение, нестационар)."""
    ss: list[float | Path]
    sy: list[float | Path]
    iconvert: int | list[int] = Field(default=0, description="0 — confined, 1 — convertible (на весь слой или список)")

    def load_arrays(self, grid):
        """Загружает ss и sy как 3D numpy-массивы."""

        def _load_list(values):
            arrays = []
            for val in values:
                if isinstance(val, (int, float)):
                    arr = np.full(grid.shape[1:], float(val))
                else:
                    arr = RasterHandler(val).resample_to_grid(grid)
                arrays.append(arr)
            return np.array(arrays)

        ss_arr = _load_list(self.ss)
        sy_arr = _load_list(self.sy)

        nlay = grid.nlay
        if isinstance(self.iconvert, int):
            iconvert_arr = np.full((nlay, *grid.shape[1:]), self.iconvert, dtype=int)
        else:
            iconvert_arr = np.array([
                np.full(grid.shape[1:], v, dtype=int)
                for v in self.iconvert
            ])

        return ss_arr, sy_arr, iconvert_arr


class FlowPackagesConfig(BaseModel):
    """Группировка всех 'flow' пакетов."""
    npf: NpfConfig | None = None
    rch: dict[int, RchConfig] | RchConfig | None = None
    ic: IcConfig | None = None
    evt: dict[int, EvtConfig] | EvtConfig | None = None
    sto: StoConfig | None = None
