# mfbuilder/mf6/mfobservations.py
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from flopy.mf6 import ModflowUtlobs

from mfbuilder.dto.observations import HeadObservation


class MF6ObservationsBuilder:
    """Создаёт единый head observation-пакет для модели MF6."""

    def __init__(self, model, grid, cfg):
        self.model = model
        self.grid = grid
        self.cfg = cfg.observations
        self._built = False

    def build(self):
        if self._built:
            return None
        self._built = True

        if not self.cfg or not self.cfg.heads:
            return None

        # ключ: имя файла вывода, значение: список записей наблюдений
        records_by_file = defaultdict(list)

        for idx, obs_def in enumerate(self.cfg.heads):
            gdf = obs_def.load_geometry(obs_def.geometry)

            # убираем NaN по уровню, но НЕ режем по году
            gdf = gdf[~gdf[obs_def.head].isna()]
            print(obs_def)
            # проверка, что столбец year есть
            if obs_def.time not in gdf.columns:
                raise ValueError("В GeoDataFrame нет столбца 'year'.")
            # группируем по году
            if obs_def.time_condition:
                gdf = gdf[gdf[obs_def.time].isin(obs_def.time_condition)]
            for year, gdf_year in gdf.groupby(obs_def.time):
                if gdf_year.empty:
                    continue

                # Имена точек
                if obs_def.name and obs_def.name in gdf_year.columns:
                    names = gdf_year[obs_def.name].astype(str).tolist()
                else:
                    # добавим год в имя, чтобы точно были уникальны
                    names = [f"obs_{idx}_{year}_{i}" for i in range(len(gdf_year))]

                # Слои
                if isinstance(obs_def.layers, int):
                    layers = [obs_def.layers] * len(gdf_year)
                elif isinstance(obs_def.layers, str) and obs_def.layers in gdf_year.columns:
                    layers = gdf_year[obs_def.layers].astype(int).tolist()
                else:
                    raise ValueError(f"Неверное значение layers: {obs_def.layers}")

                # Наблюдаемые уровни (для сравнения после расчёта)
                if isinstance(obs_def.head, (int, float)):
                    heads_obs = [float(obs_def.head)] * len(gdf_year)
                elif isinstance(obs_def.head, str) and obs_def.head in gdf_year.columns:
                    heads_obs = gdf_year[obs_def.head].astype(float).tolist()
                else:
                    heads_obs = [np.nan] * len(gdf_year)

                # Ячейки сетки
                cells = self._map_points_to_cells(gdf_year)

                year_records = []
                for name, layer, cellid, hobs in zip(names, layers, cells, heads_obs):
                    if isinstance(cellid, tuple):  # structured
                        cid = (layer - 1, *cellid)
                    else:  # vertex
                        cid = (layer - 1, cellid)
                    # (name, type, cellid, obsval)
                    year_records.append((name, "HEAD", cid, hobs))

                # Имя файла для этого года
                obs_output_filename = f"{self.model.name}.time_head_obs_{int(year)}.csv"
                records_by_file[obs_output_filename].extend(year_records)

        if not records_by_file:
            return None

        # имя для самого пакета (.obs / .pkg)
        obs_package_filename = f"{self.model.name}.head_obs_tot.obs"

        # ModflowUtlobs берёт словарь {имя_файла: список_записей}
        ModflowUtlobs(
            self.model,
            pname="head_obs",
            filename=obs_package_filename,
            print_input=True,
            digits=10,
            continuous=dict(records_by_file),
        )

        # можно вернуть список файлов, если нужно потом читать
        return list(records_by_file.keys())

    def _map_points_to_cells(self, gdf):
        from shapely.strtree import STRtree
        from shapely.geometry import Point

        if hasattr(self.grid, "geo_dataframe"):
            grid_gdf = self.grid.geo_dataframe
            tree = STRtree(grid_gdf.geometry.values)
            return [grid_gdf.index[tree.nearest(Point(p))] for p in gdf.geometry]
        else:
            raise RuntimeError("Grid не поддерживает geo_dataframe для маппинга.")

    def compare_results(self):
        """Сравнивает наблюдаемые и рассчитанные уровни (через flopy API)."""
        workspace = Path(self.model.simulation_data.mfpath.get_sim_path())
        # out_csv = workspace / f"{self.model.name}.head_obs_output.csv"
        files = workspace.glob(f"{self.model.name}.time_head_obs_*.csv")
        df_sim = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

        # 1️⃣ читаем рассчитанные уровни
        # df_sim = pd.read_csv(out_csv)
        df_sim = df_sim.melt(id_vars=["time"], var_name="name", value_name="simval")

        # 2️⃣ достаем наблюдаемые уровни из пакета наблюдений
        obs_pkg = self.model.get_package("head_obs")
        if obs_pkg is None:
            raise RuntimeError("Пакет наблюдений 'head_obs' не найден в модели.")

        obsdict = obs_pkg.continuous.get_data()
        obs_data = []
        for fileout, records in obsdict.items():
            for rec in records:
                # формат: [name, type, (lay,row,col)] или [name, type, (lay,row,col), obsval]
                name = rec[0]
                obsval = None
                if len(rec) > 3:
                    obsval = rec[3]
                obs_data.append((name, obsval))

        df_obs = pd.DataFrame(obs_data, columns=["name", "obsval"])
        df_sim["name"] = df_sim["name"].astype(str).str.strip().str.lower()
        df_obs["name"] = df_obs["name"].astype(str).str.strip().str.lower()

        # 3️⃣ объединяем по имени
        df = df_sim.merge(df_obs, on="name", how="left")
        print(df.dtypes)
        print(df.head(10))
        df["simval"] = pd.to_numeric(df["simval"], errors="coerce")
        df["obsval"] = pd.to_numeric(df["obsval"], errors="coerce")
        # 4️⃣ считаем разницу
        df["diff"] = df["simval"] - df["obsval"]
        df["abs_diff"] = df["diff"].abs()

        # 5️⃣ базовая статистика
        stats = df.dropna(subset=["obsval"]).groupby("time")["abs_diff"].agg(["mean", "max", "min"])
        print("\n📊 Ошибки по времени:")
        print(stats)

        # 6️⃣ экспорт
        df_out = workspace / f"{self.model.name}.head_obs_compare.csv"
        df.to_csv(df_out, index=False)
        print(f"\n✅ Сравнение сохранено: {df_out}")
        return df
