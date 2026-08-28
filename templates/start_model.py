import time
from mfbuilder.mfmain import LoaderYaml, Director

start = time.time()
cfg_path = '../configs/config_v1.yml'
cfg = LoaderYaml(cfg_path).read_yml()
director = Director()
ml = director.build(cfg)

# Экспорт параметров собранной модели в GIS (по одной строке на ячейку сетки,
# слои/стресс-периоды разнесены по столбцам: k_lay1, rch_sp1, chd_head_lay1_sp2 и т.д.)
# from mfbuilder.export import ModelExporter
# exporter = ModelExporter(ml)
# exporter.export_grid("../output/vectors/model_grid.geojson")
# exporter.export_package("npf", "../output/vectors/npf.geojson")

# Обратная сборка модели из такого geojson (например, после правки k в QGIS).
# geojson хранит только сетку и параметры — имя/workspace/tdis передаются отдельно.
# from mfbuilder.export import ModelImporter
# from mfbuilder.dto.base import BaseConfig, TransientConfig
# importer = ModelImporter("../output/vectors/model_grid.geojson")
# sim, model2 = importer.build(
#     base=BaseConfig(name="model_name_v2", workspace="../models/model_v2"),
#     tdis=cfg.tdis,  # или своя TransientConfig(...), если period-разбивка меняется
# )
# sim.write_simulation()
# sim.run_simulation()
