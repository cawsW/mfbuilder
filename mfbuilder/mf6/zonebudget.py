import logging
from pathlib import Path

import numpy as np

from mfbuilder.dto.zonebudget import ZoneBudgetConfig
from mfbuilder.utils.mfdata import PolygonZoneConfig


class MF6ZoneBudgetBuilder:
    """Считает баланс через MF6 ZONEBUDGET (zbud6) по уже посчитанной модели.

    Требует готовые .grb/.cbc файлы модели (запускается после builder.run()).
    Всегда сохраняет два CSV в output_dir:
      - zonebudget_zones.csv  (если задан cfg.zones) либо zonebudget_layers.csv
        (одна зона на слой, если cfg.zones не задан);
      - zonebudget_total.csv — суммарный баланс по всей активной области
        модели (idomain > 0), считается отдельным прогоном zbud6, а не
        суммированием по зонам — иначе перетоки между зонами (FROM_ZONE/
        TO_ZONE) исказили бы итог задвоенным счётом внутренних потоков.
    """

    def __init__(self, model, grid, cfg: ZoneBudgetConfig, base):
        self.model = model
        self.grid = grid
        self.cfg = cfg
        self.base = base

    def build(self) -> None:
        grb_name, cbc_name = self._package_files()
        if grb_name is None or cbc_name is None:
            logging.warning("ZoneBudget: не найдены .grb/.cbc файлы модели — пропущен.")
            return

        exe = self.cfg.exe_path or (Path(self.base.exe_path).parent / "zbud6")
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.zones is not None:
            zone_array, out_name = self._zone_array_from_file(), "zonebudget_zones.csv"
        else:
            zone_array, out_name = self._zone_array_by_layer(), "zonebudget_layers.csv"
        self._run(exe, grb_name, cbc_name, zone_array, output_dir / out_name)

        self._run(exe, grb_name, cbc_name, self._total_zone_array(), output_dir / "zonebudget_total.csv")

    # --- сборка данных ---------------------------------------------------

    def _package_files(self) -> tuple[str | None, str | None]:
        workspace = Path(self.base.workspace)
        grb_name = next(
            (f"{self.base.name}.{ext}.grb" for ext in ("disv", "dis", "disu")
             if (workspace / f"{self.base.name}.{ext}.grb").exists()),
            None,
        )
        cbc_name = f"{self.base.name}.cbb"
        if grb_name is None or not (workspace / cbc_name).exists():
            return None, None
        return grb_name, cbc_name

    def _zone_array_from_file(self) -> np.ndarray:
        zone = PolygonZoneConfig(file=self.cfg.zones, field=self.cfg.zone_field, default=0.0)
        arr2d = zone.rasterize(self.grid).astype(int)
        return np.array([arr2d for _ in range(self.grid.nlay)])

    def _zone_array_by_layer(self) -> np.ndarray:
        return np.array([np.full(self.grid.ncpl, lay + 1, dtype=int) for lay in range(self.grid.nlay)])

    def _total_zone_array(self) -> np.ndarray:
        dis_pkg = self.model.get_package("disv") or self.model.get_package("dis") or self.model.get_package("disu")
        idomain = np.asarray(dis_pkg.idomain.array)
        return np.where(idomain > 0, 1, 0).astype(int)

    def _run(self, exe, grb_name: str, cbc_name: str, zone_array: np.ndarray, out_csv: Path) -> None:
        from flopy.utils.zonbud import ZoneBudget6, ZoneFile6

        zb = ZoneBudget6(model_ws=str(self.base.workspace), exe_name=str(exe))
        zb.add_package("grb", grb_name)
        zb.add_package("bud", cbc_name)
        ZoneFile6(zb, zone_array)
        zb.write_input()
        success, buff = zb.run_model()
        if not success:
            logging.warning(f"ZoneBudget: расчёт не завершился успешно ({out_csv.name}).")
            return
        zb.get_dataframes().to_csv(out_csv)
