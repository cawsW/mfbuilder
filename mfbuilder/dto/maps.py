from typing import List, Optional, Union, Literal, Dict, Tuple, Annotated
from pydantic import BaseModel, Field


class StyleConfig(BaseModel):
    color: Optional[str] = None
    edgecolor: Optional[str] = None
    facecolor: Optional[str] = None
    linewidth: Optional[float] = None
    markersize: Optional[float] = None
    marker: Optional[str] = None
    linestyle: Optional[str] = None
    cmap: Optional[str] = None
    alpha: float = 1.0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    classification: Literal['continuous', 'quantile'] = 'continuous'
    n_classes: int = 5  # Количество интервалов для Quantile


class LabelConfig(BaseModel):
    enabled: bool = False
    column: Optional[str] = None
    fontsize: int = 8
    fontweight: str = "normal"
    color: str = "black"
    halo: bool = False
    halo_color: str = "white"
    halo_width: float = 2.0
    smart_placement: bool = True
    max_labels: int = 500


class ColorbarConfig(BaseModel):
    enabled: bool = False
    label: str = "Value"
    orientation: Literal['horizontal', 'vertical'] = 'horizontal'


class ContourStyleConfig(BaseModel):
    levels: List[float] = []
    colors: Optional[str] = "black"
    linewidths: float = 1.0
    fontsize: int = 7


class BaseLayerConfig(BaseModel):
    zorder: int = 10
    label: Optional[str] = None
    alpha: float = 1.0
    legend_patch_color: Optional[str] = None


class BasemapConfig(BaseLayerConfig):
    type: Literal['basemap']
    provider: str = "OpenStreetMap.Mapnik"
    zoom: Union[int, str] = "auto"
    zorder: int = -10


class VectorLayerConfig(BaseLayerConfig):
    type: Literal['vector']
    path: str
    filter: Optional[str] = None
    color_column: Optional[str] = None
    style: StyleConfig = Field(default_factory=StyleConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)


class AnnotationLayerConfig(BaseLayerConfig):
    type: Literal['annotation']
    path: str
    text_column: str
    color: str = "black"
    rotation: Union[float, str] = 0.0


class RasterLayerConfig(BaseLayerConfig):
    type: Literal['raster']
    path: str
    clip_by: Optional[str] = None
    style: StyleConfig = Field(default_factory=StyleConfig)
    contours: bool = False
    contour_use_head: bool = False
    contour_style: ContourStyleConfig = Field(default_factory=ContourStyleConfig)
    colorbar: ColorbarConfig = Field(default_factory=ColorbarConfig)


class FlopyLayerConfig(BaseLayerConfig):
    type: Literal['flopy']

    model_ws: str  # Путь к папке симуляции
    model_nam: Optional[str] = None  # Имя модели (если в симуляции их несколько)

    layer: int = 0  # Индекс слоя (0 - верхний)
    stress_period: int = 0  # Стресс-период (для RCH/WEL)
    parameter: Optional[str] = None  # Что рисуем: 'k', 'k33', 'rch', 'top', 'botm'

    # Настройки отображения данных (массива)
    style: StyleConfig = Field(default_factory=StyleConfig)
    log_scale: bool = False  # Логарифмическая шкала (для фильтрации)
    masked_values: List[float] = [0, 1e30, -1e30, -999.99, 999.0, -999.00, 999.99]  # Значения, считающиеся "нет данных"

    # Настройки сетки
    grid_enabled: bool = False
    grid_color: str = "gray"
    grid_linewidth: float = 0.3

    # Граничные условия (Boundary Conditions)
    # Список типов для отрисовки, например ['riv', 'drn', 'chd']
    bc_enabled: List[str] = []
    # Словарь цветов: {'riv': 'blue', 'drn': 'green'}
    bc_colors: Dict[str, str] = Field(default_factory=lambda: {'riv': 'cyan', 'drn': 'green', 'wel': 'red'})
    colorbar: ColorbarConfig = Field(default_factory=ColorbarConfig)
    contours: bool = False
    contour_style: ContourStyleConfig = Field(default_factory=ContourStyleConfig)

LayerConfigUnion = Union[BasemapConfig, VectorLayerConfig, AnnotationLayerConfig, RasterLayerConfig, FlopyLayerConfig]
LayerConfig = Annotated[LayerConfigUnion, Field(discriminator='type')]

class CrossSectionConfig(BaseModel):
    enabled: bool = False
    model_ws: Optional[str] = None
    model_nam: Optional[str] = None
    line_path: Optional[str] = None
    line_filter: Optional[str] = None
    line_label_start: str = "A"
    line_label_end: str = "B"
    show_line_on_map: bool = False
    line_color: str = "red"
    line_width: float = 1.5
    line_label_offset_points: int = 6
    parameter: Literal['head', 'k1', 'k2', 'k3'] = 'head'
    stress_period: int = 0
    style: StyleConfig = Field(default_factory=StyleConfig)
    masked_values: List[float] = [0, 1e30, -1e30, -999.99, 999.0, -999.00, 999.99]
    grid_color: str = "gray"
    grid_linewidth: float = 0.3
    contours: bool = False
    contour_style: ContourStyleConfig = Field(default_factory=ContourStyleConfig)


class MapAreaConfig(BaseModel):
    xlim: Optional[List[float]] = None
    ylim: Optional[List[float]] = None

    layers: List[LayerConfig]


class InsetMapConfig(MapAreaConfig):
    enabled: bool = False
    layers: Optional[List[LayerConfig]] = None


class SettingsConfig(BaseModel):
    title: str = "Map"
    figsize: Tuple[float, float] = (12, 8)
    output: str = "map.png"
    crs: str
    legend_loc: str = "best"
    base_fontsize: int = 10


class LegendItem(BaseModel):
    type: Literal['patch', 'line']
    color: str
    label: str


class RootConfig(BaseModel):
    settings: SettingsConfig
    main_map: MapAreaConfig
    inset_map: InsetMapConfig = Field(default_factory=InsetMapConfig)
    legend: List[LegendItem] = []
    cross_section: CrossSectionConfig = Field(default_factory=CrossSectionConfig)
