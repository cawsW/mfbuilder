import os
from pathlib import Path
import yaml
import logging
import flopy

from mfbuilder.dto.base import ProjectConfig
from mfbuilder.dto.types import EngineType
from mfbuilder.handlers import BuilderFactory, GridFactory, SourceSinksFactory, FlowParametersFactory, \
    ObservationFactory
from mfbuilder.mf6.mfmvr import MF6MvrBuilder

class LoaderYaml:
    @staticmethod
    def join_path(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*[s for s in seq])

    def __init__(self, yml):
        self.yml = yml

    def read_yml(self):
        yaml.add_constructor('!join', self.join_path)
        path = Path(self.yml).expanduser()
        raw = yaml.full_load(path.read_text())
        return ProjectConfig.model_validate(raw)


class Director:
    def __init__(self, builder_factory: BuilderFactory = BuilderFactory(),
                 grid_factory: GridFactory = GridFactory(),
                 sources_factory: SourceSinksFactory = SourceSinksFactory(),
                 flow_factory: FlowParametersFactory = FlowParametersFactory(),
                 obs_factory: ObservationFactory = ObservationFactory(),
                 mvr_builder_cls=MF6MvrBuilder,
                 ) -> None:
        self.builder_factory = builder_factory
        self.grid_factory = grid_factory
        self.sources_factory = sources_factory
        self.flow_factory = flow_factory
        self.obs_factory = obs_factory
        self.mvr_builder_cls = mvr_builder_cls

    def build(self, cfg: ProjectConfig):
        logging.basicConfig(level=logging.INFO, filename=f"{cfg.base.name}_log.log", filemode="w",
                            format="%(asctime)s %(levelname)s %(message)s")
        builder = self.builder_factory.create(cfg)
        model = builder.create_sim()
        logging.info("sim created")
        grid_obj = self.grid_factory.create(cfg)

        grid = grid_obj.create_grid(model)
        logging.info("grid created")
        pars = self.flow_factory.create(model, grid, cfg)
        pars.build()
        logging.info("flow pars created")
        observations = self.obs_factory.create(model, grid, cfg)
        observations.build()
        logging.info("obs created")
        built_packages = {}
        for pkg_name in cfg.sources.keys():
            handler = self.sources_factory.create(cfg, pkg_name, grid)
            pkg = handler.build_package(model)
            built_packages[pkg_name] = pkg
            logging.info(f"{pkg_name} created")

        if cfg.mvr and cfg.base.engine == EngineType.MF6:
            mvr_builder = self.mvr_builder_cls(model, grid, cfg.mvr, built_packages)
            mvr_builder.build()
            logging.info("mvr created")
        elif cfg.mvr:
            logging.warning("MVR поддерживается только для MF6 и будет пропущен.")

        if cfg.outputs.write_input:
            builder.finalize()
            logging.info(f"writing finish")

        if cfg.outputs.run:
            builder.run()
        # observations.compare_results()

        flopy.export.shapefile_utils.model_attributes_to_shapefile(os.path.join("..", "output", "vectors", "grid.shp"),
                                                                   ml=model)
        return model

    def load(self):
        pass

    def edit(self):
        pass
