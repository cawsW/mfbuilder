from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from mfbuilder.utils.mfdata import ParamValue, resolve_array, resolve_array_list


class NpfConfig(BaseModel):
    """Параметры пакета NPF (анизотропия, углы и т.д.)"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: list[ParamValue]
    k33overk: bool = Field(default=False, description="Использовать ли k33 как отношение к k")
    k33: list[ParamValue] | None = None
    icelltype: list[int] | int = 0

    k22overk: bool = Field(default=False, description="Использовать ли k22 как отношение к k")
    k22: list[ParamValue] | None = None

    angle1: list[ParamValue] | None = None
    angle2: list[ParamValue] | None = None
    angle3: list[ParamValue] | None = None

    def load_arrays(self, grid):
        """Загружает все параметры NPF как 3D numpy-массивы."""

        hk = resolve_array_list(self.k, grid)

        if self.k33:
            k33_arr = resolve_array_list(self.k33, grid)
            k33 = hk * k33_arr if self.k33overk else k33_arr
        else:
            k33 = hk * 0.1

        if self.k22:
            k22_arr = resolve_array_list(self.k22, grid)
            k22 = hk * k22_arr if self.k22overk else k22_arr
        else:
            k22 = hk  # по умолчанию изотропия

        angle1 = resolve_array_list(self.angle1, grid) if self.angle1 else None
        angle2 = resolve_array_list(self.angle2, grid) if self.angle2 else None
        angle3 = resolve_array_list(self.angle3, grid) if self.angle3 else None

        return hk, k22, k33, angle1, angle2, angle3


class RchConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rech: ParamValue = 0.0

    def load_array(self, grid):
        return resolve_array(self.rech, grid)


class IcConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    strt: dict[int, list[ParamValue]] | list[ParamValue] = 0.0

    def load_array(self, grid, period: int = 0):
        strt = self.strt
        if isinstance(strt, dict):
            available = sorted(strt.keys())
            # nearest lower period; if none, use the first defined
            selected = next((p for p in reversed(available) if p <= period), available[0])
            strt = strt[selected]
        return resolve_array_list(strt, grid)


class EvtConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    surface: ParamValue = 0.0
    rate: ParamValue = 0.001
    depth: ParamValue = 1.0
    ievt: ParamValue | None = None

    def load_arrays(self, grid):
        surface = resolve_array(self.surface, grid)
        rate = resolve_array(self.rate, grid)
        depth = resolve_array(self.depth, grid)
        ievt = resolve_array(self.ievt, grid).astype(int) if self.ievt is not None else None
        return surface, rate, depth, ievt


class StoConfig(BaseModel):
    """Параметры пакета STO (хранение, нестационар)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ss: list[ParamValue]
    sy: list[ParamValue]
    iconvert: int | list[int] = Field(default=0, description="0 — confined, 1 — convertible (на весь слой или список)")

    def load_arrays(self, grid):
        """Загружает ss и sy как 3D numpy-массивы."""

        ss_arr = resolve_array_list(self.ss, grid)
        sy_arr = resolve_array_list(self.sy, grid)

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
