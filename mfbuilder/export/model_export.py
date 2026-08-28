import re
from pathlib import Path

import numpy as np


class ModelExporter:
    """Экспортирует параметры уже собранной/загруженной MF6-модели в GIS-формат.

    Работает как с моделью, только что построенной через Director.build(cfg)
    (она возвращает flopy-модель), так и с моделью, загруженной с диска через
    from_workspace(). Не зависит от ProjectConfig/Director — это отдельная,
    более поздняя стадия работы с моделью (интроспекция уже существующей
    flopy-модели), поэтому вынесена в отдельный модуль, а не встроена в Director.

    Два способа экспорта:
      - export_grid(path)     — вся сетка со всеми параметрами в один файл,
                                 по одной строке на уникальную ячейку (node);
                                 слои/стресс-периоды разносятся по столбцам.
      - export_package(name, path) — один пакет (npf, chd, riv, ghb, ...)
                                 в отдельный файл, той же логикой именования
                                 столбцов.

    Именование столбцов:
      - послойные параметры (npf/ic/sto, top/botm): "{имя}_lay{N}", слои с 1.
      - инфильтрация (rcha): "rch_sp{P}", стресс-периоды с 1.
      - списочные ГУ-пакеты (chd/riv/ghb/drn/wel/...): "{пакет}_{поле}_lay{N}_sp{P}".
        Разбивки по временным шагам (timestep) нет — входные данные MODFLOW
        (граничные условия, питание) заданы на stress period целиком, шаг
        на них не влияет.
      - столбец с пустыми значениями (комбинация слой/период без данных в
        этом пакете) не создаётся вовсе, чтобы не раздувать таблицу.
    """

    # Пакеты, у которых нужные данные — обычные послойные массивы (не списки
    # по ячейкам). Ключ — имя пакета в нижнем регистре, значение — атрибуты,
    # которые нужно вытащить (каждый как массив (nlay, ...) или (...,) для 1 слоя).
    LAYERED_ARRAY_PACKAGES = {
        "npf": ["k", "k22", "k33", "angle1", "angle2", "angle3", "icelltype"],
        "ic": ["strt"],
        "sto": ["ss", "sy", "iconvert"],
    }

    def __init__(self, model):
        self.model = model
        self.grid = model.modelgrid
        self._base_gdf = None

    @classmethod
    def from_workspace(cls, sim_ws, model_nam: str | None = None, crs: str | None = None) -> "ModelExporter":
        """Загружает модель с диска (как flopy.mf6.MFSimulation.load) и оборачивает в ModelExporter.

        MF6 не хранит CRS в своих файлах, поэтому после загрузки с диска
        model.modelgrid.crs обычно None — передайте crs (например "EPSG:3857",
        как в grid.epsg главного конфига), чтобы он попал в экспортируемый файл.
        """
        import flopy

        sim = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws))
        model = sim.get_model(model_nam) if model_nam else sim.get_model()
        if crs is not None:
            model.modelgrid.crs = crs
        return cls(model)

    # --- публичное API -------------------------------------------------

    def export_grid(self, path, stress_periods: list[int] | None = None):
        """Экспортирует сетку со всеми параметрами модели в один файл (geojson/gpkg/shp)."""
        gdf = self._new_gdf()

        self._add_top_botm(gdf)
        self._add_idomain(gdf)
        for pkg_name, attrs in self.LAYERED_ARRAY_PACKAGES.items():
            pkg = self.model.get_package(pkg_name)
            if pkg is not None:
                self._add_array_package(pkg, attrs, gdf)
        self._add_recharge(gdf, stress_periods)

        for pkg_name in self.model.get_package_list():
            pkg = self.model.get_package(pkg_name)
            if pkg is not None and hasattr(pkg, "stress_period_data"):
                self._add_list_package(pkg, self._clean_pkg_name(pkg_name), gdf, stress_periods)

        self._write(gdf, path)
        return gdf

    def export_package(self, package_name: str, path, stress_periods: list[int] | None = None):
        """Экспортирует один пакет модели (например 'npf', 'chd', 'riv', 'rcha')."""
        pkg = self.model.get_package(package_name)
        if pkg is None:
            raise ValueError(
                f"Пакет '{package_name}' не найден в модели. Доступны: {self.model.get_package_list()}"
            )

        gdf = self._new_gdf()
        clean_name = self._clean_pkg_name(package_name)

        if clean_name == "rcha":
            self._add_recharge(gdf, stress_periods)
        elif hasattr(pkg, "stress_period_data"):
            self._add_list_package(pkg, clean_name, gdf, stress_periods)
        elif clean_name in self.LAYERED_ARRAY_PACKAGES:
            self._add_array_package(pkg, self.LAYERED_ARRAY_PACKAGES[clean_name], gdf)
        else:
            raise ValueError(
                f"Автоматический экспорт пакета '{package_name}' не поддерживается "
                "(неизвестная структура данных — не массив по слоям и не список по ячейкам)."
            )

        self._write(gdf, path)
        return gdf

    # --- сборка данных ---------------------------------------------------

    def _new_gdf(self):
        if self._base_gdf is None:
            self._base_gdf = self.grid.geo_dataframe
        return self._base_gdf.copy()

    def _flatten(self, arr) -> np.ndarray:
        return np.asarray(arr).reshape(-1)

    def _add_top_botm(self, gdf):
        gdf["top"] = self._flatten(self.grid.top)
        botm = np.asarray(self.grid.botm)
        for lay in range(self.grid.nlay):
            gdf[f"botm_lay{lay + 1}"] = self._flatten(botm[lay])

    def _add_idomain(self, gdf):
        dis_pkg = self.model.get_package("disv") or self.model.get_package("dis") or self.model.get_package("disu")
        if dis_pkg is None or getattr(dis_pkg, "idomain", None) is None:
            return
        idomain = np.asarray(dis_pkg.idomain.array)
        for lay in range(self.grid.nlay):
            gdf[f"idomain_lay{lay + 1}"] = self._flatten(idomain[lay]).astype(int)

    def _add_array_package(self, pkg, attrs, gdf):
        for attr in attrs:
            data = getattr(pkg, attr, None)
            if data is None:
                continue
            arr = getattr(data, "array", data)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            for lay in range(arr.shape[0]):
                gdf[f"{attr}_lay{lay + 1}"] = self._flatten(arr[lay])

    def _add_recharge(self, gdf, stress_periods=None):
        pkg = self.model.get_package("rcha")
        if pkg is None:
            return
        arr = np.asarray(pkg.recharge.array)  # (nper, 1, ncpl) либо (nper, 1, nrow, ncol)
        nper = arr.shape[0]
        periods = stress_periods if stress_periods is not None else range(nper)
        for sp in periods:
            if sp >= nper:
                continue
            gdf[f"rch_sp{sp + 1}"] = self._flatten(arr[sp, 0])

    def _add_list_package(self, pkg, pkg_name, gdf, stress_periods=None):
        data_dict = pkg.stress_period_data.get_data()
        if data_dict is None:
            return
        if not isinstance(data_dict, dict):
            data_dict = {0: data_dict}

        periods = stress_periods if stress_periods is not None else sorted(data_dict.keys())
        nlay = self.grid.nlay
        ncells = len(gdf)
        structured = self.grid.grid_type == "structured"
        ncol = self.grid.ncol if structured else None

        for sp in periods:
            rec = data_dict.get(sp)
            if rec is None or len(rec) == 0:
                continue
            fields = [f for f in rec.dtype.names if f != "cellid"]
            # Числовые поля — float64 с NaN (GeoJSON пишет их как числа); текстовые
            # (boundname и т.п.) — object с None. Смешанный dtype=object для всех
            # полей writer сериализует как строки даже для чисел — этого избегаем.
            numeric_field = {field: np.issubdtype(rec.dtype[field], np.number) for field in fields}
            buffers = {
                (field, lay): np.full(ncells, np.nan) if numeric_field[field] else np.full(ncells, None, dtype=object)
                for field in fields for lay in range(nlay)
            }

            # Несколько исходных объектов могут попадать в одну и ту же ячейку/слой
            # (например, два экрана скважины в одной ячейке) — таблица "1 строка = 1
            # ячейка" не может хранить их отдельно. Числовые поля в этом случае
            # суммируются (ровно так их эффективно и складывает сам MODFLOW6 при
            # решении — несколько записей WEL/RIV/... в одной ячейке работают как
            # параллельные независимые условия).
            seen_cells = set()
            collisions = 0
            for row in rec:
                cellid = row["cellid"]
                lay = cellid[0]
                flat_idx = cellid[1] * ncol + cellid[2] if structured else cellid[1]
                if (lay, flat_idx) in seen_cells:
                    collisions += 1
                seen_cells.add((lay, flat_idx))
                for field in fields:
                    if numeric_field[field]:
                        cur = buffers[(field, lay)][flat_idx]
                        buffers[(field, lay)][flat_idx] = row[field] if np.isnan(cur) else cur + row[field]
                    else:
                        buffers[(field, lay)][flat_idx] = row[field]

            if collisions:
                print(
                    f"ModelExporter: пакет '{pkg_name}', период {sp + 1} — {collisions} запис(ей) "
                    "делят ячейку/слой с другой записью того же пакета; числовые поля просуммированы."
                )

            for (field, lay), values in buffers.items():
                empty = np.isnan(values).all() if numeric_field[field] else all(v is None for v in values)
                if empty:
                    continue
                gdf[f"{pkg_name}_{field}_lay{lay + 1}_sp{sp + 1}"] = values

    # --- вспомогательное ---------------------------------------------------

    @staticmethod
    def _clean_pkg_name(name: str) -> str:
        """'CHD_0' -> 'chd'. Числовой суффикс flopy добавляет даже для единственного
        экземпляра пакета — mfbuilder всегда строит по одному пакету на тип, поэтому
        обрезать суффикс безопасно и даёт более читаемые имена столбцов."""
        return re.sub(r"_\d+$", "", name).lower()

    def _write(self, gdf, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if gdf.crs is None and getattr(self.grid, "crs", None) is not None:
            gdf = gdf.set_crs(self.grid.crs)
        driver = "GeoJSON" if path.suffix.lower() in (".geojson", ".json") else None
        gdf.to_file(path, driver=driver)
