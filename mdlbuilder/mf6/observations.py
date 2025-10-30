# mdlbuilder/mf6/mfobservations.py
from pathlib import Path
import pandas as pd
import numpy as np
from flopy.mf6 import ModflowUtlobs


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

        all_records = []
        for idx, obs_def in enumerate(self.cfg.heads):
            gdf = obs_def.load_geometry(obs_def.geometry)

            # Имена точек
            if obs_def.name and obs_def.name in gdf.columns:
                names = gdf[obs_def.name].astype(str).tolist()
            else:
                names = [f"obs_{idx}_{i}" for i in range(len(gdf))]

            # Слои
            if isinstance(obs_def.layers, int):
                layers = [obs_def.layers] * len(gdf)
            elif isinstance(obs_def.layers, str) and obs_def.layers in gdf.columns:
                layers = gdf[obs_def.layers].astype(int).tolist()
            else:
                raise ValueError(f"Неверное значение layers: {obs_def.layers}")

            # Наблюдаемые уровни (для сравнения после расчёта)
            if isinstance(obs_def.head, (int, float)):
                heads_obs = [float(obs_def.head)] * len(gdf)
            elif isinstance(obs_def.head, str) and obs_def.head in gdf.columns:
                heads_obs = gdf[obs_def.head].astype(float).tolist()
            else:
                heads_obs = [np.nan] * len(gdf)

            # Ячейки сетки
            cells = self._map_points_to_cells(gdf)

            for name, layer, cellid, hobs in zip(names, layers, cells, heads_obs):
                if isinstance(cellid, tuple):  # structured
                    cid = (layer - 1, *cellid)
                else:  # vertex
                    cid = (layer - 1, cellid)
                all_records.append((name, "HEAD", cid, hobs))

        # ✅ 1. имя для самого пакета (в .nam)
        obs_package_filename = f"{self.model.name}.head_obs.pkg"

        # ✅ 2. отдельный файл, в который MF6 будет писать результаты
        obs_output_filename = f"{self.model.name}.head_obs_output.csv"

        ModflowUtlobs(
            self.model,
            pname="head_obs",
            filename=obs_package_filename,  # .pkg
            print_input=True,
            digits=10,
            continuous={obs_output_filename: all_records},
        )

        return obs_output_filename

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
        out_csv = workspace / f"{self.model.name}.head_obs_output.csv"

        # 1️⃣ читаем рассчитанные уровни
        df_sim = pd.read_csv(out_csv)
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
