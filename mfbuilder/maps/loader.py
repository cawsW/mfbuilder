from typing import Optional
from mfbuilder.maps.protocols import IMapLayer
from mfbuilder.maps.layers import VectorLayer, RasterLayer, BasemapLayer, FlopyLayer, AnnotationLayer
from mfbuilder.dto.maps import LayerConfig

class LayerFactory:
    @staticmethod
    def create_layer(layer_conf: LayerConfig, global_crs: Optional[str] = None) -> IMapLayer:
        if layer_conf.type == 'vector':
            return VectorLayer(layer_conf, global_crs)
        elif layer_conf.type == 'raster':
            return RasterLayer(layer_conf, global_crs)
        elif layer_conf.type == 'basemap':
            return BasemapLayer(layer_conf, global_crs)
        elif layer_conf.type == 'flopy':
            return FlopyLayer(layer_conf, global_crs)
        elif layer_conf.type == 'annotation':
            return AnnotationLayer(layer_conf, global_crs)
        else:
            raise ValueError(f"Unknown layer type: {layer_conf.type}")
