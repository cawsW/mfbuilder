import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from flopy.mf6 import (
    MFSimulation, ModflowGwf, ModflowIms, ModflowTdis, ModflowGwfdisv, ModflowGwfoc,
    ModflowGwfnpf, ModflowGwfic, ModflowGwfsto, ModflowGwfrcha,
    ModflowGwfchd, ModflowGwfriv, ModflowGwfghb, ModflowGwfdrn, ModflowGwfwel, ModflowGwfrch,
)
from flopy.utils.cvfdutil import to_cvfd, get_disv_gridprops

from mfbuilder.dto.base import BaseConfig, TransientConfig
from mfbuilder.dto.types import EngineType

# Пакеты списочных ГУ, которые ModelImporter умеет собирать обратно из
# столбцов "{pkg}_{field}_lay{N}_sp{P}". Добавьте сюда свой пакет, если
# используете его и хотите, чтобы он восстанавливался автоматически.
BC_PACKAGE_CLASSES = {
    "chd": ModflowGwfchd,
    "riv": ModflowGwfriv,
    "ghb": ModflowGwfghb,
    "drn": ModflowGwfdrn,
    "wel": ModflowGwfwel,
    "rch": ModflowGwfrch,
}

_LAYERED_RE = re.compile(r"^[a-zA-Z0-9]+_lay(\d+)$")
_RCH_SP_RE = re.compile(r"^rch_sp(\d+)$")
_BC_RE = re.compile(r"^(?P<pkg>[a-zA-Z0-9]+)_(?P<field>[a-zA-Z0-9]+)_lay(?P<lay>\d+)_sp(?P<sp>\d+)$")

# Совпадает с MF6Builder.create_ims() / default_ims_kwargs() в mfbuilder.mf6.mfbuilder.
# Не импортируется оттуда напрямую: mfbuilder.mf6.mfbuilder <-> mfbuilder.mfmain
# <-> mfbuilder.handlers образуют цикл, который сегодня "работает" только благодаря
# порядку импортов внутри mfmain.py — заходить в этот цикл со стороны mf6.mfbuilder
# при первом импорте не безопасно.
_IMS_KWARGS = dict(
    complexity="COMPLEX",
    outer_maximum=500,
    outer_dvclose=1e-4,
    inner_maximum=100,
    inner_dvclose=1e-4,
    under_relaxation="DBD",
    linear_acceleration="BICGSTAB",
)


class ModelImporter:
    """Строит MF6-модель заново из geojson, экспортированного ModelExporter.export_grid().

    Зеркало ModelExporter в обратную сторону. Столбцы geojson однозначно
    определяют: геометрию сетки (DISV — по одной ячейке на строку), top/botm,
    idomain, NPF (k/k22/k33/angle*/icelltype), IC (strt), STO (ss/sy/iconvert),
    RCHA (rch_spN) и списочные ГУ-пакеты (chd/riv/ghb/drn/wel/...) по столбцам
    вида "{pkg}_{field}_lay{N}_sp{P}".

    Чего в geojson нет и что нужно передать отдельно в build(): имя модели,
    рабочая директория, exe, единицы времени (BaseConfig) и разбивка по
    stress period — perlen/nstp/tsmult/steady (TransientConfig). Экспортёр их
    не хранит, так как это настройки симуляции, а не пространственные данные.

    Поддерживается только сетка type: vertex (DISV) — её строят все текущие
    проекты. Для structured/unstructured геометрии восстановление не
    реализовано (нет надёжного способа проверить его на реальных данных).

    Не восстанавливается (ModelExporter это и не экспортирует): LAK, MVR.
    """

    def __init__(self, path):
        self.path = Path(path)
        gdf = gpd.read_file(self.path)
        if "node" not in gdf.columns:
            raise ValueError(
                f"В {self.path} нет столбца 'node' — похоже, файл не является "
                "экспортом ModelExporter.export_grid()."
            )
        if "row" in gdf.columns and "col" in gdf.columns:
            raise NotImplementedError(
                "ModelImporter пока умеет восстанавливать только вертексную сетку (DISV). "
                f"{self.path} содержит структурированную сетку (есть столбцы row/col)."
            )
        self.gdf = gdf.sort_values("node").reset_index(drop=True)
        self.ncpl = len(self.gdf)
        self.nlay = self._detect_nlay()

    def _detect_nlay(self) -> int:
        layers = {int(m.group(1)) for col in self.gdf.columns if (m := _LAYERED_RE.match(col))}
        if not layers:
            raise ValueError("Не удалось определить число слоёв — нет столбцов вида '..._lay{N}'.")
        return max(layers)

    # --- публичное API -------------------------------------------------

    def build(self, base: BaseConfig, tdis: TransientConfig | None = None):
        """Собирает MFSimulation + ModflowGwf. Не пишет и не запускает модель —
        вызовите sim.write_simulation() / sim.run_simulation() сами (как в
        MF6Builder.finalize()/run())."""
        if base.engine != EngineType.MF6:
            raise ValueError("ModelImporter поддерживает только engine=mf6.")
        tdis = tdis or TransientConfig()

        sim = MFSimulation(sim_name=base.name, version="mf6", exe_name=base.exe_path, sim_ws=str(base.workspace))
        ModflowTdis(sim, nper=tdis.nper, time_units=base.tunits.value, perioddata=tdis.perioddata)
        ModflowIms(sim, **_IMS_KWARGS)
        model = ModflowGwf(sim, modelname=base.name, save_flows=True, newtonoptions="NEWTON")

        self._build_disv(model)
        self._build_npf(model)
        self._build_ic(model)
        self._build_sto(model, tdis)
        self._build_rcha(model)
        self._build_bc_packages(model)

        ModflowGwfoc(
            model,
            pname="oc",
            budget_filerecord=f"{base.name}.cbb",
            head_filerecord=f"{base.name}.hds",
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )

        return sim, model

    # --- сетка -----------------------------------------------------------

    @staticmethod
    def _to_float(series) -> np.ndarray:
        """Числовые столбцы geojson (в т.ч. после ручной правки в QGIS) не всегда
        читаются geopandas как float — приводим явно вместо .to_numpy(dtype=...)."""
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    def _layered_array(self, prefix, dtype=float):
        cols = [f"{prefix}_lay{i + 1}" for i in range(self.nlay)]
        if not all(c in self.gdf.columns for c in cols):
            return None
        arr = np.array([self._to_float(self.gdf[c]) for c in cols])
        return arr if dtype is float else arr.astype(dtype)

    def _build_disv(self, model):
        vertdict = {}
        for i, geom in enumerate(self.gdf.geometry):
            coords = list(geom.exterior.coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            vertdict[i] = coords
        verts, iverts = to_cvfd(vertdict)
        gridprops = get_disv_gridprops(verts, iverts)

        top = self._to_float(self.gdf["top"])
        botm = self._layered_array("botm")
        if botm is None:
            raise ValueError("В geojson нет столбцов 'botm_lay{N}' — сетку нельзя восстановить.")
        idomain = self._layered_array("idomain", dtype=int)

        ModflowGwfdisv(
            model,
            nlay=self.nlay,
            top=top,
            botm=botm,
            idomain=idomain if idomain is not None else 1,
            **gridprops,
        )

    # --- послойные пакеты --------------------------------------------------

    def _build_npf(self, model):
        k = self._layered_array("k")
        if k is None:
            raise ValueError("В geojson нет столбцов 'k_lay{N}' — пакет NPF нельзя восстановить.")
        icelltype = self._layered_array("icelltype", dtype=int)
        ModflowGwfnpf(
            model,
            icelltype=icelltype if icelltype is not None else 0,
            k=k,
            k22=self._layered_array("k22"),
            k33=self._layered_array("k33"),
            angle1=self._layered_array("angle1"),
            angle2=self._layered_array("angle2"),
            angle3=self._layered_array("angle3"),
        )

    def _build_ic(self, model):
        strt = self._layered_array("strt")
        if strt is not None:
            ModflowGwfic(model, strt=strt)

    def _build_sto(self, model, tdis: TransientConfig):
        ss = self._layered_array("ss")
        sy = self._layered_array("sy")
        if ss is None or sy is None:
            return
        iconvert = self._layered_array("iconvert", dtype=int)

        steady_list = tdis.steady
        if isinstance(steady_list, bool):
            steady_list = [steady_list]
        steady_state = {i: True for i, s in enumerate(steady_list) if s}
        transient = {i: True for i, s in enumerate(steady_list) if not s}

        ModflowGwfsto(
            model,
            iconvert=iconvert if iconvert is not None else 0,
            ss=ss,
            sy=sy,
            steady_state=steady_state or None,
            transient=transient or None,
        )

    def _build_rcha(self, model):
        recharge = {}
        for col in self.gdf.columns:
            m = _RCH_SP_RE.match(col)
            if m:
                sp = int(m.group(1)) - 1
                recharge[sp] = self._to_float(self.gdf[col])
        if recharge:
            ModflowGwfrcha(model, readasarrays=True, recharge=recharge)

    # --- списочные ГУ-пакеты ------------------------------------------------

    def _build_bc_packages(self, model):
        grouped: dict[str, list[tuple[str, str, int, int]]] = {}
        for col in self.gdf.columns:
            m = _BC_RE.match(col)
            if not m:
                continue
            grouped.setdefault(m.group("pkg"), []).append(
                (col, m.group("field"), int(m.group("lay")), int(m.group("sp")))
            )

        for pkg_name, entries in grouped.items():
            pkg_cls = BC_PACKAGE_CLASSES.get(pkg_name)
            if pkg_cls is None:
                print(
                    f"ModelImporter: пропущен пакет '{pkg_name}' — неизвестный тип "
                    f"(добавьте его в mfbuilder.export.model_import.BC_PACKAGE_CLASSES)."
                )
                continue
            self._build_one_bc_package(model, pkg_name, pkg_cls, entries)

    def _build_one_bc_package(self, model, pkg_name, pkg_cls, entries):
        has_boundname = any(field == "boundname" for _, field, _, _ in entries)
        periods = sorted({sp for _, _, _, sp in entries})
        stress_period_data = {}

        for sp in periods:
            sp_entries = [e for e in entries if e[3] == sp]
            fields_by_layer: dict[int, dict[str, str]] = {}
            for col, field, lay, _ in sp_entries:
                fields_by_layer.setdefault(lay, {})[field] = col

            records = []
            for lay, field_cols in fields_by_layer.items():
                value_fields = [f for f in field_cols if f != "boundname"]
                field_arrays = {f: self._to_float(self.gdf[field_cols[f]]) for f in value_fields}

                valid_mask = np.ones(self.ncpl, dtype=bool)
                for arr in field_arrays.values():
                    valid_mask &= ~np.isnan(arr)

                bname_col = field_cols.get("boundname")
                bname_arr = self.gdf[bname_col].to_numpy() if bname_col else None

                for idx in np.nonzero(valid_mask)[0]:
                    record = [(lay - 1, int(idx))] + [field_arrays[f][idx] for f in value_fields]
                    if has_boundname:
                        bname = bname_arr[idx] if bname_arr is not None else None
                        if bname is None or (isinstance(bname, float) and np.isnan(bname)):
                            bname = ""
                        record.append(str(bname))
                    records.append(record)

            stress_period_data[sp - 1] = records

        kwargs = {"stress_period_data": stress_period_data, "pname": pkg_name}
        if has_boundname:
            kwargs["boundnames"] = True
        pkg_cls(model, **kwargs)
