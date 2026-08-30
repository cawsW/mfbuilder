import shutil
from pathlib import Path

import numpy as np

from mfbuilder.dto.pest import PestWorkspaceConfig, ParameterGroupConfig


class WorkspacePreparer:
    """Готовит одноразовую рабочую копию модели для PstFrom.

    - копирует model_ws -> calib_ws (оригинал никогда не трогается — как во
      всех текущих pest_*.py скриптах: model_ws остаётся источником истины,
      calib_ws одноразовый и пересоздаётся при каждой сборке);
    - кладёт в неё нужные исполняемые файлы (mf6, pestpp-*, zbud6...);
    - разворачивает многострочные текстовые файлы массивов в один столбец —
      MF6 пишет их с переносом строк на фиксированной ширине (число значений
      в строке может отличаться от строки к строке для нестандартных ncpl),
      а pyemu умеет параметризовать только построчный (одно значение — одна
      строка) формат.
    """

    def __init__(self, cfg: PestWorkspaceConfig):
        self.cfg = cfg

    def prepare(self, parameter_groups: list[ParameterGroupConfig]) -> Path:
        self._copy_workspace()
        self._copy_executables()
        if self.cfg.flatten_arrays:
            self._flatten_array_files(parameter_groups)
        return self.cfg.calib_ws

    def _copy_workspace(self) -> None:
        if not self.cfg.model_ws.exists():
            raise FileNotFoundError(f"Директория модели не найдена: {self.cfg.model_ws}")
        if self.cfg.calib_ws.exists():
            shutil.rmtree(self.cfg.calib_ws)
        shutil.copytree(self.cfg.model_ws, self.cfg.calib_ws)

    def _copy_executables(self) -> None:
        for exe in self.cfg.exe_paths:
            shutil.copy2(exe, self.cfg.calib_ws / exe.name)

    def _flatten_array_files(self, parameter_groups: list[ParameterGroupConfig]) -> None:
        for group in parameter_groups:
            if group.index_cols:
                continue  # списочный файл (stress_period_data) — уже построчный формат
            for filename in group.files:
                self._flatten_one(self.cfg.calib_ws / filename)

    @staticmethod
    def _flatten_one(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Файл параметра не найден в рабочей копии: {path}")
        values = path.read_text().split()
        arr = np.array(values, dtype=float).reshape(-1, 1)
        np.savetxt(path, arr, fmt="%.8e")
