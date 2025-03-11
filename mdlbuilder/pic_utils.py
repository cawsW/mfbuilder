import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import geopandas as gpd
import contextily as ctx
import os
from shapely.geometry import Point, LineString, Polygon
from matplotlib import patches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
import random
import numpy as np
import mapclassify
from textwrap import wrap

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math

# Попытка импортировать adjustText (для раздвигания подписей)
try:
    from adjustText import adjust_text

    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False

# Импорт для работы с растровыми данными
import rasterio
from rasterio.mask import mask
import rasterio.plot

# Глобальная конфигурация шрифта
GLOBAL_FONT = {"family": "Arial", "size": 9}
GLOBAL_TEXT_COLOR = "black"


def get_font_prop():
    """Возвращает объект FontProperties на основе GLOBAL_FONT."""
    return FontProperties(**GLOBAL_FONT)


# --------- 0. Функция для добавления масштабной линейки ---------
def add_scalebar(ax, scale_text=None, location=None, length_fraction=0.2, fontdict=None):
    """
    Рисует масштабную линейку на оси ax.

    Если scale_text не передан, он вычисляется динамически на основе экстента карты.
    Масштабная линейка центрируется по горизонтали.
    """
    if location is None:
        location = (0.5 - length_fraction / 2, 0.05)
    if fontdict is None:
        fontdict = {**GLOBAL_FONT, "color": GLOBAL_TEXT_COLOR}

    # Получаем экстент оси
    x0_data, x1_data = ax.get_xlim()
    data_width = x1_data - x0_data
    bar_data_length = data_width * length_fraction

    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    axis_width_pixels = bbox.width
    bar_pixel_length = axis_width_pixels * length_fraction

    dpi = ax.figure.dpi
    bar_length_inches = bar_pixel_length / dpi
    bar_length_meters = bar_length_inches * 0.0254  # 1 дюйм = 0.0254 м

    scale_ratio = bar_data_length / bar_length_meters if bar_length_meters else 1
    computed_scale_text = f"1:{int(round(scale_ratio))}"
    if scale_text is None:
        scale_text = computed_scale_text

    ax.hlines(y=location[1],
              xmin=location[0],
              xmax=location[0] + length_fraction,
              transform=ax.transAxes,
              colors='black',
              linewidth=2)
    ax.text(location[0] + length_fraction / 2,
            location[1] - 0.03,
            scale_text,
            transform=ax.transAxes,
            ha='center',
            va='top',
            **fontdict)


# --------- 1. Класс для описания векторного слоя (Layer) ---------
class Layer:
    """
    Описывает векторный слой на карте.

    :param path: путь к файлу (shp/geojson и т.п.)
    :param label: подпись для легенды
    :param color: основной цвет для отрисовки
    :param edgecolor: цвет границы (для полигонов)
    :param linewidth: толщина линий или границ
    :param marker: маркер для точек
    :param markersize: размер маркера
    :param alpha: прозрачность
    :param fill: если False, для полигонов заливка отключается (facecolor="none")
    :param label_field: имя столбца, содержащего подписи для точечных объектов
    :param label_fontdict: шрифт для подписей; если не задан – используется GLOBAL_FONT и GLOBAL_TEXT_COLOR
    :param label_buffer: базовое смещение подписи
    :param show_in_legend: отображать ли слой в легенде
    """

    def __init__(self,
                 path,
                 label,
                 color='blue',
                 edgecolor='black',
                 linewidth=1.0,
                 marker='o',
                 markersize=5,
                 alpha=1,
                 fill=True,
                 label_field=None,
                 label_fontdict=None,
                 label_buffer=10,
                 show_in_legend=True):
        self.path = path
        self.label = label
        self.color = color
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.marker = marker
        self.markersize = markersize
        self.alpha = alpha
        self.fill = fill
        self.label_field = label_field
        self.label_fontdict = label_fontdict if label_fontdict is not None else {**GLOBAL_FONT,
                                                                                 "color": GLOBAL_TEXT_COLOR}
        self.label_buffer = label_buffer
        self.show_in_legend = show_in_legend


class QuantilePointLayer(Layer):
    """
    Отображает точки из шейп-файла с квантильной классификацией по заданному атрибуту.

    :param attribute: имя столбца для классификации
    :param num_quantiles: количество квантилей (например, 5 или 7)
    :param cmap: colormap для раскрашивания точек
    :param legend_label: подпись для легенды (для отображения диапазонов значений)
    """

    def __init__(self, path, label, attribute, num_quantiles=5, cmap='viridis',
                 legend_label=None, **kwargs):
        super().__init__(path, label, **kwargs)
        self.attribute = attribute
        self.num_quantiles = num_quantiles
        self.cmap = cmap
        self.legend_label = legend_label if legend_label is not None else label
        self._mappable = None
        self._bins = None

    def draw(self, ax, crs=3857):
        gdf = gpd.read_file(self.path)
        if gdf.crs is not None:
            gdf = gdf.to_crs(epsg=crs)
        classifier = mapclassify.Quantiles(gdf[self.attribute], k=self.num_quantiles)
        # classifier = mapclassify.FisherJenksSampled(gdf[self.attribute], k=self.num_quantiles)
        bins = np.concatenate(([gdf[self.attribute].min()], classifier.bins))
        norm = plt.Normalize(vmin=bins[0], vmax=bins[-1])
        cmap_obj = plt.get_cmap(self.cmap)
        mappable = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf[self.attribute],
                              cmap=cmap_obj, norm=norm, s=self.markersize,
                              alpha=self.alpha, edgecolor=self.edgecolor, zorder=5)
        self._mappable = mappable
        self._bins = bins
        return mappable

# --------- 2. Класс для растрового слоя (RasterLayer) ---------
class RasterLayer:
    """
    Отображает растровый слой на карте.

    :param raster_path: путь к файлу растра (например, GeoTIFF)
    :param cmap: colormap для отображения растра
    :param alpha: прозрачность растра
    :param show_contours: флаг, указывающий, нужно ли наносить изолинии
    :param contour_levels: список уровней для изолиний; если None – вычисляются автоматически
    :param clip_polygon: путь к шейп-файлу с полигоном для обрезки или объект геометрии/список геометрий.
    :param log_transform: bool, если True, значения растра логарифмируются (np.log)
    :param add_colorbar: bool, если True, под картой по центру отображается colorbar
    :param colorbar_label: строка для подписи цветовой легенды (colorbar)
    """

    def __init__(self, raster_path, cmap='viridis', alpha=1.0,
                 show_contours=False, contour_levels=None, clip_polygon=None,
                 log_transform=False, add_colorbar=True, colorbar_label="Value", inline_spacing=5):
        self.raster_path = raster_path
        self.cmap = cmap
        self.alpha = alpha
        self.show_contours = show_contours
        self.contour_levels = contour_levels
        self.clip_polygon = clip_polygon
        self.log_transform = log_transform
        self.add_colorbar = add_colorbar
        self.colorbar_label = colorbar_label
        self.inline_spacing = inline_spacing

    def get_nice_levels(self, data, lower_quantile=0.05, upper_quantile=0.95, n_ticks=10):
        # Проверка, что значения квантилей корректны
        if not (0 <= lower_quantile <= 1 and 0 <= upper_quantile <= 1):
            raise ValueError("Параметры lower_quantile и upper_quantile должны находиться в диапазоне [0, 1]")

        # Если в данных могут быть NaN, используем np.nanquantile
        qmin = np.nanquantile(data, lower_quantile)
        qmax = np.nanquantile(data, upper_quantile)
        raw_range = qmax - qmin

        if raw_range <= 0:
            return np.array([qmin])

        if raw_range > 1:
            raw_step = raw_range / float(n_ticks)
            exponent = math.floor(math.log10(raw_step))
            fraction = raw_step / (10 ** exponent)
            if fraction < 1.5:
                nice_fraction = 1.0
            elif fraction < 3:
                nice_fraction = 2.0
            elif fraction < 7:
                nice_fraction = 5.0
            else:
                nice_fraction = 10.0
            step = nice_fraction * (10 ** exponent)
            lower_candidate = math.floor(qmin / step) * step
            upper_candidate = math.ceil(qmax / step) * step
            # Если разница между квантилью и кандидатом мала, используем сам квантиль
            if abs(qmin - lower_candidate) < 0.2 * step:
                lower_bound = qmin
            else:
                lower_bound = lower_candidate
            if abs(upper_candidate - qmax) < 0.2 * step:
                upper_bound = qmax
            else:
                upper_bound = upper_candidate
            levels = np.arange(lower_bound, upper_bound + step / 2, step)
        else:
            step = 0.05
            levels = np.arange(qmin, round(qmax + step / 2, 2), step)
        return levels

    def draw(self, ax):
        with rasterio.open(self.raster_path) as src:
            if self.clip_polygon is not None:
                if isinstance(self.clip_polygon, str):
                    clip_gdf = gpd.read_file(self.clip_polygon)
                    clip_gdf = clip_gdf.to_crs(src.crs)
                    clip_geom = clip_gdf.geometry.values.tolist()
                elif isinstance(self.clip_polygon, list):
                    clip_geom = self.clip_polygon
                else:
                    clip_geom = [self.clip_polygon]
                data, transform = mask(src, clip_geom, crop=True)
                data = data[0]
            else:
                data = src.read(1, masked=True)
                transform = src.transform
            # Если log_transform включён, применяем логарифмическое преобразование
            if self.log_transform:
                # Для безопасности, заменяем не положительные значения на nan
                data = np.where(data > 0, data, np.nan)
                data = np.log10(data)
            from rasterio.plot import show, plotting_extent
            im = show(data, transform=transform, ax=ax, cmap=self.cmap, alpha=self.alpha)
            extent = plotting_extent(src) if self.clip_polygon is None else plotting_extent(data, transform=transform)
        if self.show_contours:
            if self.contour_levels is None:
                levels = self.get_nice_levels(data)
            else:
                levels = self.contour_levels
            contour = show(data, transform=transform, ax=ax, contour=True, levels=levels,
                           colors="black", linewidths=0.5, contour_label_kws={"fontsize": 6,
                                                                              "inline_spacing": self.inline_spacing})


        if self.add_colorbar:
            # Добавляем цветовую легенду под картой (горизонтальный colorbar)
            im_g = im.get_images()[0]
            cbar = plt.colorbar(im_g, ax=ax, orientation='horizontal', shrink=0.4, pad=0.01)
            cbar.ax.set_xlabel(self.colorbar_label, fontdict={**GLOBAL_FONT, "color": GLOBAL_TEXT_COLOR})
            cbar.ax.tick_params(labelsize=8)
        return im


class PolygonGridLayer:
    """
    Отображает слой с полигональной сеткой, раскрашивая полигоны в соответствии с указанным атрибутом.

    :param path: путь к шейп-файлу с полигонами сетки
    :param attribute: имя столбца для раскрашивания
    :param cmap: colormap для раскрашивания полигонов
    :param alpha: прозрачность слоя
    :param edgecolor: цвет границы полигонов
    :param linewidth: толщина границ
    :param legend_label: подпись для цветовой шкалы
    """

    def __init__(self, path, attribute, cmap='viridis', alpha=0.8, edgecolor='black', linewidth=0.5,
                 legend_label="Value",
                 log_transform=False, clip_polygon=None):
        self.path = path
        self.attribute = attribute
        self.cmap = cmap
        self.alpha = alpha
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.legend_label = legend_label
        self.log_transform = log_transform
        self.clip_polygon = gpd.read_file(clip_polygon)

    def draw(self, ax, crs=3857):
        gdf = gpd.read_file(self.path)
        if gdf.crs:
            gdf = gdf.to_crs(epsg=crs)
        if self.clip_polygon is not None:
            gdf = gpd.clip(gdf, self.clip_polygon)

        data = gdf[self.attribute].copy()
        if self.log_transform:
            data = np.where(data > 0, np.log10(data), np.nan)

        gdf = gdf.assign(plot_attr=data)
        plot_grd = gdf.plot(column='plot_attr', ax=ax, cmap=self.cmap, edgecolor=self.edgecolor,
                            linewidth=self.linewidth, alpha=self.alpha, legend=True, k=10,
                            legend_kwds={'label': self.legend_label, 'orientation': 'horizontal', 'shrink': 0.65,
                                         'pad': 0.01})

        # Настройка шрифтов для легенды
        colorbar = plot_grd.get_figure().axes[-1]
        colorbar.set_xlabel(self.legend_label, fontdict={**GLOBAL_FONT, "color": GLOBAL_TEXT_COLOR})
        colorbar.tick_params(labelsize=8)


# --------- 3. Базовый класс для карты (BaseMap) ---------
class BaseMap:
    """
    Отвечает за отрисовку слоёв на оси Axes:
      - Чтение векторных данных, перевод в заданный CRS (по умолчанию EPSG:3857)
      - Отрисовка векторных слоёв (точки, линии, полигоны) с аннотациями
      - Отрисовка растровых слоёв (если заданы) поверх подложки или вместо нее
      - Добавление масштабной линейки, рамки, настройка экстента
      - Возможность "раздвигать" подписи точек (avoid_label_overlap)
    """

    def __init__(self, layers, raster_layers=None, with_basemap=False, extent=None, crs=3857,
                 avoid_label_overlap=False):
        """
        :param layers: список векторных слоёв (Layer)
        :param raster_layers: список растровых слоёв (RasterLayer), по умолчанию None
        :param with_basemap: bool, флаг подложки OSM (применяется для векторных данных)
        :param extent: [xmin, ymin, xmax, ymax] или None
        :param crs: EPSG-код, в который переводятся векторные данные
        :param avoid_label_overlap: bool, использовать ли алгоритм раздвигания подписей
        """
        self.layers = layers
        self.raster_layers = raster_layers if raster_layers is not None else []
        self.with_basemap = with_basemap
        self.extent = extent
        self.crs = crs
        self.avoid_label_overlap = avoid_label_overlap
        self._text_objects = []

    def draw(self, ax):
        # Сначала отрисовываем растровые слои (фон)
        for r_layer in self.raster_layers:
            r_layer.draw(ax)

        # Затем отрисовываем векторные слои
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        all_bounds = []
        for layer in self.layers:
            gdf = self._read_and_reproject(layer)
            if not gdf.empty:
                all_bounds.append(gdf.total_bounds)
            if isinstance(layer, QuantilePointLayer):
                layer.draw(ax, crs=self.crs)
            else:
                self._draw_layer(ax, layer, gdf)

        if self.extent is not None:
            ax.set_xlim(self.extent[0], self.extent[2])
            ax.set_ylim(self.extent[1], self.extent[3])
        else:
            if all_bounds:
                xmin = min(b[0] for b in all_bounds)
                ymin = min(b[1] for b in all_bounds)
                xmax = max(b[2] for b in all_bounds)
                ymax = max(b[3] for b in all_bounds)
                ax.set_xlim(xmin - 2750, xmax + 2671.8)
                ax.set_ylim(ymin - 2750, ymax + 2671.8)
        if self.with_basemap:
            ctx.add_basemap(ax, attribution='', crs=self.crs, alpha=0.4, source=ctx.providers.OpenStreetMap.Mapnik,
                            zorder=0)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)

        # add_scalebar(ax, scale_text=None, location=None, length_fraction=0.2)

        # if self.avoid_label_overlap and ADJUST_TEXT_AVAILABLE and self._text_objects:
        #     adjust_text(
        #         self._text_objects,
        #         ax=ax,
        #         autoalign='xy', expand_objects=(0.1, 1),
        #         only_move={'points': '', 'text': 'y', 'objects': 'y'},
        #         force_text=0.75, force_objects=0.1,
        #         arrowprops=dict(arrowstyle="simple, head_width=0.25, tail_width=0.05", color='r', lw=0.5, alpha=0.5)
        #     )

    def _read_and_reproject(self, layer):
        gdf = gpd.read_file(layer.path, encoding="utf-8")
        if gdf.crs is not None:
            gdf = gdf.to_crs(epsg=self.crs)
        return gdf

    def _draw_layer(self, ax, layer, gdf):
        if gdf.empty:
            return
        geom_type = gdf.geom_type.iloc[0].lower()
        if 'point' in geom_type:
            # Пример фильтрации: можно оставить только точки с определённым атрибутом (пример: gdf["lay"]==3)
            gdf.plot(ax=ax,
                     marker=layer.marker,
                     markersize=layer.markersize,
                     edgecolor=layer.edgecolor,
                     color=layer.color,
                     alpha=layer.alpha)
            if layer.label_field:
                self._annotate_points(ax, gdf, layer, self.avoid_label_overlap)
        elif 'line' in geom_type:
            gdf.plot(ax=ax,
                     color=layer.color,
                     linewidth=layer.linewidth,
                     alpha=layer.alpha)
        elif 'polygon' in geom_type:
            facecolor = layer.color if layer.fill else "none"
            gdf.plot(ax=ax,
                     facecolor=facecolor,
                     edgecolor=layer.edgecolor,
                     linewidth=layer.linewidth,
                     alpha=layer.alpha)
            if layer.label_field:
                self._annotate_points(ax, gdf, layer, self.avoid_label_overlap)
        else:
            gdf.plot(ax=ax,
                     color=layer.color,
                     edgecolor=layer.edgecolor,
                     linewidth=layer.linewidth,
                     alpha=layer.alpha)

    def get_candidate_offsets(self, buff):
        """
        Генерирует список вариантов смещения с выравниванием для заданного buff.
        dx, dy ∈ {buff, 0, -buff} для каждого, формируются все 9 вариантов.
        Здесь каждый кандидат определяется автоматически: если dx>0 → ha='left', dx<0 → ha='right', dx==0 → ha='center';
        аналогично для dy и va.
        """
        candidates = []
        for dx in [buff, 0, -buff]:
            for dy in [buff, 0, -buff]:
                ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
                va = 'bottom' if dy > 0 else ('top' if dy < 0 else 'center')
                candidates.append((dx, dy, ha, va))
        return candidates

    def _annotate_points(self, ax, gdf, layer, avoid):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        max_allowed_buffer = min((x1 - x0) / 2, (y1 - y0) / 2)
        renderer = ax.figure.canvas.get_renderer()
        placed_extents = []
        threshold_distance = 350  # пороговое расстояние для корректировки смещения

        geom_type = gdf.geom_type.iloc[0].lower()

        for idx, row in gdf.iterrows():
            if row.geometry is not None:
                if 'polygon' in geom_type:
                    x, y = row.geometry.centroid.x, row.geometry.centroid.y
                else:
                    x, y = row.geometry.x, row.geometry.y
                label_text = str(row[layer.label_field])
                if not avoid:
                    text_obj = ax.annotate(
                        label_text,
                        xy=(x, y),
                        xytext=(layer.label_buffer, layer.label_buffer),
                        textcoords='offset points',
                        ha="center",
                        va="center",
                        path_effects=[pe.withStroke(linewidth=1, foreground="white")],
                        **layer.label_fontdict,
                        transform=ax.transData
                    )
                else:
                    distances = []
                    for idx2, row2 in gdf.iterrows():
                        if idx != idx2 and row2.geometry is not None:
                            distances.append(row.geometry.distance(row2.geometry))
                    min_distance = min(distances) if distances else threshold_distance
                    factor = 1.0
                    if min_distance < threshold_distance:
                        factor = 1 + (threshold_distance - min_distance) / threshold_distance
                    random_factor = random.uniform(0.8, 1.6)
                    effective_buffer = layer.label_buffer * factor * random_factor

                    candidate_found = False
                    current_buffer = effective_buffer
                    chosen_text_obj = None
                    chosen_extent = None
                    max_iterations = 50
                    iter_count = 0

                    while not candidate_found and iter_count < max_iterations:
                        if current_buffer > max_allowed_buffer:
                            break
                        candidate_offsets = self.get_candidate_offsets(current_buffer)
                        random.shuffle(candidate_offsets)
                        for dx, dy, ha, va in candidate_offsets:
                            arrowprops = dict(
                                arrowstyle="->",
                                color='black',
                                lw=0.5,
                                shrinkA=0.1,
                                shrinkB=0.9
                            )
                            text_obj = ax.annotate(
                                label_text,
                                xy=(x, y),
                                xytext=(dx, dy),
                                textcoords='offset points',
                                ha=ha,
                                va=va,
                                arrowprops=arrowprops,
                                path_effects=[pe.withStroke(linewidth=1, foreground="white")],
                                **layer.label_fontdict,
                                transform=ax.transData
                            )
                            extent = text_obj.get_window_extent(renderer=renderer)
                            if not any(extent.overlaps(prev) for prev in placed_extents):
                                chosen_text_obj = text_obj
                                chosen_extent = extent
                                candidate_found = True
                                break
                            else:
                                text_obj.remove()
                        if not candidate_found:
                            current_buffer *= 1.5
                            iter_count += 1
                    if candidate_found:
                        placed_extents.append(chosen_extent)


# --------- 4. Класс для легенды ---------
class Legend:
    """
    Отрисовывает легенду на оси на основе списка слоев.
    Если слой является экземпляром QuantilePointLayer, строит вертикальную легенду,
    отображающую точки разными цветами с диапазонами значений.
    """
    def __init__(self, layers, inner=True):
        self.layers = layers
        self.inner = inner

    def draw(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        if not self.inner:
            for spine in ax.spines.values():
                spine.set_visible(False)

        handles = []
        labels = []
        quantile_layers = []
        for layer in self.layers:
            if not getattr(layer, 'show_in_legend', True):
                continue
            if isinstance(layer, QuantilePointLayer):
                quantile_layers.append(layer)
                continue
            # Формирование символов для обычных слоев
            gdf = gpd.read_file(layer.path, encoding="utf-8")
            if gdf.empty:
                continue
            geom_type = gdf.geom_type.iloc[0].lower()
            if 'point' in geom_type:
                handle = Line2D([0], [0],
                                marker=layer.marker,
                                markersize=10,
                                markeredgecolor=layer.edgecolor,
                                color=layer.color,
                                linestyle='None',
                                alpha=layer.alpha)
            elif 'line' in geom_type:
                handle = Line2D([0, 1], [0, 0],
                                color=layer.color,
                                linewidth=layer.linewidth,
                                alpha=layer.alpha)
            elif 'polygon' in geom_type:
                handle = patches.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=layer.color if layer.fill else "none",
                    edgecolor=layer.edgecolor,
                    linewidth=layer.linewidth,
                    alpha=layer.alpha)
            else:
                handle = Line2D([0], [0],
                                marker=layer.marker,
                                markersize=layer.markersize,
                                color=layer.color,
                                linestyle='None',
                                alpha=layer.alpha)
            handles.append(handle)
            labels.append(layer.label)
        # Если имеются QuantilePointLayer, создаем для них отдельную легенду
        if quantile_layers:
            # Предположим, что используется только один QuantilePointLayer
            q_layer = quantile_layers[0]
            if q_layer._bins is not None:
                bins = q_layer._bins
                cmap_obj = plt.get_cmap(q_layer.cmap)
                norm = plt.Normalize(vmin=bins[0], vmax=bins[-1])
                dummy_handle = Line2D([], [], linestyle='None')
                handles.append(dummy_handle)
                labels.append(q_layer.legend_label)
                for i in range(len(bins)-1):
                    mid = (bins[i] + bins[i+1]) / 2
                    color = cmap_obj(norm(mid))
                    handle = Line2D([0], [0], marker='o', markersize=10,
                                    color=color, linestyle='None', markeredgecolor=q_layer.edgecolor)
                    label_str = f"{bins[i]:.2f} - {bins[i+1]:.2f}"
                    handles.append(handle)
                    labels.append(label_str)
        if handles:
            labels = [labels[0]] + ['\n'.join(wrap(l, 30)) for l in labels[1:]]
            leg = ax.legend(
                handles, labels,
                loc= "upper right" if self.inner else "upper left",
                prop=get_font_prop(),
                frameon=True,
                fancybox=True
            )
            leg.set_title(
                "Условные обозначения",
                prop={**GLOBAL_FONT, "weight": "bold"},
                # pad=2
            )
            # Компоновка: легенда в правом верхнем углу карты
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_alpha(0.7)
            leg.get_frame().set_linewidth(0)
        if not self.inner:
            ax.axis('off')

# --------- 5. Класс для компоновки (MapLayout) ---------
class MapLayout:
    def __init__(self, main_map: BaseMap, legend: Legend, path, sub_map: BaseMap = None):
        self.main_map = main_map
        self.legend = legend
        self.sub_map = sub_map
        self.path = path

    def render(self):
        # if self.sub_map:
        #     fig = plt.figure(figsize=(12, 8))
        #     gs = GridSpec(nrows=2, ncols=2,
        #                   width_ratios=[3, 1.2],
        #                   height_ratios=[2, 1.3],
        #                   figure=fig)
        #     ax_main = fig.add_subplot(gs[:, 0])
        #     ax_sub = fig.add_subplot(gs[0, 1])
        #     ax_legend = fig.add_subplot(gs[1, 1])
        #     self.main_map.draw(ax_main)
        #     self.sub_map.draw(ax_sub)
        #     self.legend.draw(ax_legend)
        #     plt.tight_layout()
        #     # gs.update(wspace=0.0, left=0.0, right=0.6, top=1, bottom=0.0)
        # else:
        #     fig = plt.figure(figsize=(12, 8))
        #     gs = GridSpec(nrows=1, ncols=2,
        #                   width_ratios=[3, 0.8],
        #                   figure=fig)
        #     ax_main = fig.add_subplot(gs[0, 0])
        #     ax_legend = fig.add_subplot(gs[0, 1])
        #     self.main_map.draw(ax_main)
        #     self.legend.draw(ax_legend)
        #     gs.update(wspace=0.0, left=0.0, right=0.6, top=1, bottom=0.0)
        # plt.savefig(self.path, dpi=300,
        #             bbox_inches='tight')

        if self.sub_map:
            # Оригинальная компоновка с субкартой
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(nrows=2, ncols=2,
                          width_ratios=[3, 1.2],
                          height_ratios=[2, 1.3],
                          figure=fig)
            ax_main = fig.add_subplot(gs[:, 0])
            ax_sub = fig.add_subplot(gs[0, 1])
            ax_legend = fig.add_subplot(gs[1, 1])
            self.main_map.draw(ax_main)
            self.sub_map.draw(ax_sub)
            self.legend.draw(ax_legend)
            plt.tight_layout()
        else:
            if self.legend.inner:
                fig, ax_main = plt.subplots(figsize=(12, 12))
                self.main_map.draw(ax_main)
                self.legend.draw(ax_main)
                plt.tight_layout()
            else:
                fig = plt.figure(figsize=(12, 8))
                gs = GridSpec(nrows=1, ncols=2,
                              width_ratios=[3, 0.8],
                              figure=fig)
                ax_main = fig.add_subplot(gs[0, 0])
                ax_legend = fig.add_subplot(gs[0, 1])
                self.main_map.draw(ax_main)
                self.legend.draw(ax_legend)
                gs.update(wspace=0.0, left=0.0, right=0.6, top=1, bottom=0.0)
        plt.savefig(self.path, dpi=300, bbox_inches='tight')
        # plt.show()

# --------- 6. Пример использования ---------
