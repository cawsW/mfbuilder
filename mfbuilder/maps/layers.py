import os
from typing import Any, List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import contextily as ctx
import flopy
import pyemu
from rasterio.mask import mask
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as patheffects
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

from mfbuilder.maps.protocols import IMapLayer
from mfbuilder.dto.maps import VectorLayerConfig, RasterLayerConfig, BasemapConfig, FlopyLayerConfig, AnnotationLayerConfig, PestLayerConfig


class BasemapLayer(IMapLayer):
    def __init__(self, config: BasemapConfig, global_crs: Optional[str] = None):
        super().__init__(config, global_crs)
        self.config: BasemapConfig = config

    def draw(self, ax: plt.Axes) -> None:
        provider_name = self.config.provider
        zoom = self.config.zoom
        alpha = self.config.alpha

        try:
            source = ctx.providers
            for part in provider_name.split('.'):
                source = source.get(part)
                if source is None:
                    source = provider_name
                    break
        except Exception:
            source = provider_name

        if self.global_crs:
            try:
                ctx.add_basemap(
                    ax,
                    crs=self.global_crs,
                    source=source,
                    alpha=alpha,
                    zoom=zoom,
                    zorder=self.config.zorder,
                    attribution=''
                )
                custom_text = 'EPSG:2497'
                ax.text(
                    0.98,
                    0.02,
                    custom_text,
                    transform=ax.transAxes,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=12,
                    color='black'
                )
            except Exception as e:
                print(f"Error adding basemap: {e}")
        else:
            print("Error: Global CRS is not set.")

    def get_legend_handles(self) -> List[Any]:
        return []


class VectorLayer(IMapLayer):
    def __init__(self, config: VectorLayerConfig, global_crs: Optional[str] = None):
        super().__init__(config, global_crs)
        self.config: VectorLayerConfig = config
        self._detected_geom_type: Optional[str] = None
        self._legend_items: List[Dict[str, Any]] = []

    def draw(self, ax: plt.Axes) -> None:
        path = self.config.path

        try:
            self._legend_items = []
            gdf = self._load_and_process_data()
            if gdf is None or gdf.empty:
                return
            if not gdf.empty and self._detected_geom_type is None:
                types = gdf.geom_type.unique()
                if any('Polygon' in t for t in types):
                    self._detected_geom_type = 'polygon'
                elif any('Line' in t for t in types):
                    self._detected_geom_type = 'line'
                else:
                    self._detected_geom_type = 'point'

            plot_kwargs = self._prepare_style(ax, gdf)
            gdf.plot(**plot_kwargs)

            if self.config.labels.enabled:
                self._add_labels(ax, gdf, zorder=self.config.zorder + 1)

        except Exception as e:
            print(f"Failed to process vector layer {self.config.path}: {e}")

    def _load_and_process_data(self) -> Optional[gpd.GeoDataFrame]:
        path = self.config.path
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            print(f"Error reading vector file {path}: {e}")
            return None

        if self.global_crs:
            gdf = gdf.set_crs(self.global_crs, allow_override=True)

        if self.config.filter:
            gdf = self._apply_filter(gdf)

        return gdf

    def _apply_filter(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        query = self.config.filter
        try:
            col, val = query.split('=', 1)
            if col not in gdf.columns:
                print(f"Warning: Filter column '{col}' not found.")
                return gdf

            if gdf[col].dtype == 'O':
                return gdf[gdf[col] == val]
            else:
                try:
                    num_val = float(val)
                    return gdf[gdf[col] == num_val]
                except ValueError:
                    return gdf[gdf[col] == val]
        except ValueError:
            return gdf

    def _prepare_style(self, ax: plt.Axes, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        style = self.config.style
        kwargs = {
            'ax': ax,
            'zorder': self.config.zorder,
            'label': self.config.label,
            'alpha': style.alpha
        }

        if self.config.color_column and self.config.color_column in gdf.columns:
            column = self.config.color_column
            valid_values = gdf[column].dropna().values
            if valid_values.size > 0:
                vmin = style.vmin if style.vmin is not None else float(np.min(valid_values))
                vmax = style.vmax if style.vmax is not None else float(np.max(valid_values))
                bins = self._build_bins(valid_values, vmin, vmax, style.classification, style.n_classes)
                cmap_name = style.cmap or 'viridis'
                if bins is not None and len(bins) > 1:
                    cmap = plt.get_cmap(cmap_name, len(bins) - 1)
                    norm = BoundaryNorm(boundaries=bins, ncolors=len(bins) - 1)
                    kwargs['column'] = column
                    kwargs['cmap'] = cmap
                    kwargs['norm'] = norm
                    self._legend_items = self._build_legend_items(bins, cmap)
                    kwargs['label'] = '_nolegend_'
                else:
                    kwargs['column'] = column
                    kwargs['cmap'] = cmap_name
                    kwargs['vmin'] = vmin
                    kwargs['vmax'] = vmax

        if style.color and 'column' not in kwargs: kwargs['color'] = style.color
        if style.edgecolor: kwargs['edgecolor'] = style.edgecolor
        if style.linewidth: kwargs['linewidth'] = style.linewidth
        if style.markersize: kwargs['markersize'] = style.markersize
        if style.marker: kwargs['marker'] = style.marker
        if style.linestyle: kwargs['linestyle'] = style.linestyle
        if style.facecolor and 'column' not in kwargs: kwargs['facecolor'] = style.facecolor
        if style.cmap and 'cmap' not in kwargs: kwargs['cmap'] = style.cmap
        if 'color' not in kwargs and 'facecolor' not in kwargs and 'column' not in kwargs and not style.cmap:
            kwargs['color'] = 'blue'

        return kwargs

    @staticmethod
    def _build_bins(values: np.ndarray, vmin: float, vmax: float, classification: str, n_classes: int) -> Optional[np.ndarray]:
        if n_classes < 1:
            return None
        if classification == 'quantile':
            bins = np.percentile(values, np.linspace(0, 100, n_classes + 1))
            bins = np.unique(bins)
            return bins if len(bins) > 1 else None
        if vmin == vmax:
            return None
        return np.linspace(vmin, vmax, n_classes + 1)

    def _build_legend_items(self, bins: np.ndarray, cmap) -> List[Dict[str, Any]]:
        items = []
        for idx in range(len(bins) - 1):
            color = cmap(idx)
            start = bins[idx]
            end = bins[idx + 1]
            items.append({'label': f"{start:.3g} - {end:.3g}", 'color': color})
        return items

    def get_legend_handles(self) -> List[Any]:
        """
        Генерирует элементы легенды на основе РЕАЛЬНОГО типа геометрии.
        """
        if self._legend_items:
            handles = []
            style = self.config.style
            marker = style.marker or 'o'
            # marker_size = (style.markersize or 6) / 2.5
            marker_size = 10
            if self.config.label:
                header = Line2D([0], [0], color='none', ls='', label=self.config.label)
                setattr(header, '_mfbuilder_force_legend', True)
                handles.append(header)
            for item in self._legend_items:
                handle = Line2D(
                    [0], [0],
                    marker=marker,
                    color='none',
                    ls='',
                    markerfacecolor=item['color'],
                    markeredgecolor=style.edgecolor or 'black',
                    markersize=marker_size,
                    label=item['label']
                )
                setattr(handle, '_mfbuilder_force_legend', True)
                handles.append(handle)
            return handles

        label = self.config.label
        if not label:
            return []

        style = self.config.style

        color = style.color or 'blue'
        edgecolor = style.edgecolor or color
        facecolor = style.facecolor or color

        geom_type = self._detected_geom_type

        if geom_type == 'polygon':
            return [Patch(
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=style.linewidth or 0.5,
                alpha=style.alpha,
                label=label
            )]

        elif geom_type == 'line':
            return [Line2D(
                [0], [0],
                color=color,
                linewidth=style.linewidth or 1.5,
                alpha=style.alpha,
                linestyle=style.linestyle,
                label=label
            )]

        elif geom_type == 'point':
            return [Line2D(
                [0], [0],
                marker=style.marker or 'o',
                color=None,
                ls='',
                markerfacecolor=color,
                markeredgecolor=style.edgecolor or 'white',
                markersize=(style.markersize or 5) * 0.2,
                alpha=style.alpha,
                label=label
            )]

        else:
            if style.linewidth is not None and style.markersize is None:
                return [Line2D([0], [0], color=color, lw=style.linewidth,
                               label=label)]
            return [Patch(facecolor=facecolor, label=label)]

    def _add_labels(self, ax: plt.Axes, gdf: gpd.GeoDataFrame, zorder: int):
        lbl_conf = self.config.labels
        column = lbl_conf.column
        if not column or column not in gdf.columns:
            return

        try:
            renderer = ax.figure.canvas.get_renderer()
        except Exception:
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()

        occupied_bboxes = []

        for _, row in gdf.iterrows():
            geom = row.geometry
            if not geom or geom.is_empty: continue
            p = geom.centroid
            pxy = ax.transData.transform((p.x, p.y))
            occupied_bboxes.append(mtransforms.Bbox.from_bounds(pxy[0] - 2, pxy[1] - 2, 4, 4))

        data_to_label = gdf.head(lbl_conf.max_labels)

        for _, row in data_to_label.iterrows():
            geom = row.geometry
            if not geom: continue
            p = geom.centroid

            label_text = str(row[column])

            # Позиции
            POSITION_MAP = {
                'top': ('center', 'bottom', 0, 5),
                'bottom': ('center', 'top', 0, -5),
                'left': ('right', 'center', -5, 0),
                'right': ('left', 'center', 5, 0),
                'top-left': ('right', 'bottom', -5, 5),
                'top-right': ('left', 'bottom', 5, 5),
                'bottom-left': ('right', 'top', -5, -5),
                'bottom-right': ('left', 'top', 5, -5),
                'center': ('center', 'center', 0, 0),
            }

            # 1. Формируем список кандидатов на основе конфига
            if getattr(lbl_conf, 'auto_placement', True):
                placements = [
                    ('left', 'bottom', 5, 5),
                    ('right', 'bottom', -5, 5),
                    ('center', 'top', 0, -5),
                    ('center', 'bottom', 0, 5)
                ]
            else:
                # Берем конкретную позицию из конфига (по умолчанию 'top')
                pos_key = getattr(lbl_conf, 'position', 'top')
                placements = [POSITION_MAP.get(pos_key, ('center', 'top', 0, -5))]

            # 2. Отрисовка
            for ha, va, off_x, off_y in placements:
                t_obj = ax.annotate(
                    label_text, xy=(p.x, p.y), xytext=(off_x, off_y),
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=lbl_conf.fontsize, fontweight=lbl_conf.fontweight,
                    color=lbl_conf.color, zorder=zorder, clip_on=True
                )

                bbox = t_obj.get_window_extent(renderer).expanded(1.05, 1.05)

                # Если авто-подбор отключен, считаем, что перекрытия нет
                if not getattr(lbl_conf, 'auto_placement', True):
                    overlap = False
                else:
                    overlap = any(bbox.overlaps(o) for o in occupied_bboxes)

                if not overlap:
                    occupied_bboxes.append(bbox)
                    if getattr(lbl_conf, 'halo', False):
                        t_obj.set_path_effects([
                            pe.withStroke(linewidth=lbl_conf.halo_width, foreground=lbl_conf.halo_color),
                            pe.Normal()
                        ])
                    break
                else:
                    t_obj.remove()


class RasterLayer(IMapLayer):
    def __init__(self, config: RasterLayerConfig, global_crs: Optional[str] = None):
        super().__init__(config, global_crs)
        self.config: RasterLayerConfig = config

    @staticmethod
    def _generate_coordinates(h, w, transform):
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        xs, ys = rasterio.transform.xy(transform, list(rows.flatten()), list(cols.flatten()), offset='center')
        return np.array(xs).reshape(h, w), np.array(ys).reshape(h, w)

    @staticmethod
    def _apply_nodata_mask(data, nodata_val):
        if nodata_val is None: nodata_val = -99999
        if np.isnan(nodata_val): return np.ma.masked_invalid(data)
        return np.ma.masked_equal(data, nodata_val)

    def draw(self, ax: plt.Axes) -> None:
        path = self.config.path

        try:
            with rasterio.open(path) as src:
                img_data, transform = self._load_data(src)
                if img_data is None: return

                img_data = self._apply_nodata_mask(img_data, src.nodata)

                self._draw_raster(ax, img_data, transform)

                if self.config.contours:
                    self._draw_contours(ax, img_data, transform)

        except Exception as e:
            print(f"Failed to process raster layer {path}: {e}")

    def _load_data(self, src):
        clip_path = self.config.clip_by
        if not clip_path:
            return src.read(1), src.transform
        try:
            clip_gdf = gpd.read_file(clip_path)
            if self.global_crs:
                clip_gdf = clip_gdf.set_crs(self.global_crs, allow_override=True)

            geoms = clip_gdf.geometry.values
            nodata = src.nodata if src.nodata is not None else -99999
            out_image, out_transform = mask(src, geoms, crop=True, nodata=nodata)
            return out_image[0], out_transform
        except Exception as e:
            print(f"Clip error: {e}")
            return None, None

    def _draw_raster(self, ax, data, transform):
        style = self.config.style
        extent = plotting_extent(data, transform)
        self.mappable = ax.imshow(
            data, extent=extent,
            cmap=style.cmap or 'viridis',
            alpha=style.alpha,
            vmin=style.vmin, vmax=style.vmax
        )

    def _draw_contours(self, ax, data, transform):
        c_style = self.config.contour_style
        h, w = data.shape
        xs, ys = self._generate_coordinates(h, w, transform)

        cnt = ax.contour(xs, ys, data,
                         levels=c_style.levels,
                         colors=c_style.colors,
                         linewidths=c_style.linewidths)
        ax.clabel(cnt, cnt.levels, inline=True, fontsize=c_style.fontsize)

    def get_legend_handles(self) -> List[Any]:
        return []


class AnnotationLayer(IMapLayer):
    def __init__(self, config: AnnotationLayerConfig, global_crs: Optional[str] = None):
        super().__init__(config, global_crs)
        self.config: AnnotationLayerConfig = config

    def draw(self, ax: plt.Axes) -> None:
        path = self.config.path
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            print(f"Error reading annotation file {path}: {e}")
            return

        if self.global_crs:
            gdf = gdf.set_crs(self.global_crs, allow_override=True)

        if self.config.text_column not in gdf.columns:
            print(f"Warning: Annotation column '{self.config.text_column}' not found in {path}.")
            return

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == 'Point':
                x, y = geom.x, geom.y
            else:
                centroid = geom.centroid
                x, y = centroid.x, centroid.y

            rotation = self._resolve_rotation(row)
            ax.text(
                x,
                y,
                str(row[self.config.text_column]),
                color=self.config.color,
                rotation=rotation,
                zorder=self.config.zorder,
                fontdict={'fontsize': self.config.fontsize},
                path_effects = [patheffects.withStroke(linewidth=2, foreground='white')]
            )

    def _resolve_rotation(self, row: gpd.GeoSeries) -> float:
        rotation = self.config.rotation
        if isinstance(rotation, str):
            if rotation in row.index:
                value = row.get(rotation)
                try:
                    return float(value) if value is not None else 0.0
                except (TypeError, ValueError):
                    return 0.0
            try:
                return float(rotation)
            except ValueError:
                return 0.0
        return float(rotation)

    def get_legend_handles(self) -> List[Any]:
        return []


class FlopyCrossSection:
    _model_cache = {}

    def __init__(self, config, global_crs: Optional[str] = None):
        self.config = config
        self.global_crs = global_crs
        self.model = self._load_model()
        self._line_coords: Optional[List[Tuple[float, float]]] = None
        self.legend_handles: List[Any] = []

    def draw(self, ax: plt.Axes) -> None:
        if not self.model:
            return

        line = self._load_line_coords()
        if not line:
            return
        pmv = flopy.plot.PlotCrossSection(model=self.model, line={'line': line}, ax=ax, geographic_coords=True)
        data = self._get_array_data()
        if data is None:
            return

        style = self.config.style
        cmap = plt.get_cmap(style.cmap or 'viridis')

        pa = pmv.plot_array(
            data,
            head=data,
            cmap=cmap,
            vmin=style.vmin,
            vmax=style.vmax,
            alpha=style.alpha,
            masked_values=self.config.masked_values
        )
        head = self._get_heads_data()
        if not self.config.layers:
            nlay = self.model.modelgrid.nlay
            colors = plt.cm.Spectral(np.linspace(0, 1, nlay))
        else:
            nlay = self.config.layers
            colors = plt.cm.Spectral(np.linspace(0, 1, len(nlay)))
        self._plot_head_surfaces_smooth(pmv, head, ax, nlay, colors)
        pmv.plot_grid(color=self.config.grid_color, linewidth=self.config.grid_linewidth)
        # plt.colorbar(pa, shrink=0.75)

        if self.config.contours:
            c_style = self.config.contour_style
            levels = c_style.levels
            if not levels:
                levels = self._calculate_auto_levels(data)
            try:
                cs = pmv.contour_array(
                    data,
                    head=data,
                    masked_values=self.config.masked_values,
                    levels=levels,
                    colors=c_style.colors,
                    linewidths=c_style.linewidths
                )
                if cs is not None:
                    ax.clabel(cs, inline=True, fontsize=c_style.fontsize)
            except RuntimeError as e:
                print(f"Warning: contour_array failed: {e}")

        self._label_section_ends(ax)
        self._plot_surface_raster(ax)
        self._apply_section_ylim(ax)

    def _plot_head_surfaces_smooth(self, pmv, head, ax, nlay, colors):
        self.legend_handles = []
        seq = list(range(nlay)) if type(nlay) is int else nlay
        for idx, i in enumerate(seq):
            label = f"Поверхность подземных вод {i + 1} слоя (на разрезе)"
            color = colors[idx]

            # Запоминаем количество artist'ов ДО вызова plot_surface
            n_lines_before = len(ax.lines)
            n_colls_before = len(ax.collections)

            pmv.plot_surface(head[i], color=color, lw=0.1)

            # Забираем всё что добавил flopy
            new_lines = list(ax.lines[n_lines_before:])
            new_colls = list(ax.collections[n_colls_before:])

            xs, ys = [], []

            for line in new_lines:
                try:
                    xd = np.asarray(line.get_xdata(), dtype=float)
                    yd = np.asarray(line.get_ydata(), dtype=float)
                    if len(xd) >= 2 and len(yd) >= 1:
                        xs.append(float(np.mean(xd)))
                        ys.append(float(yd[0]))
                except Exception:
                    pass

            for coll in new_colls:
                try:
                    for seg in coll.get_segments():
                        seg = np.asarray(seg, dtype=float)
                        if seg.shape[0] >= 2:
                            xs.append(float(np.mean(seg[:, 0])))
                            ys.append(float(seg[0, 1]))
                except Exception:
                    pass

            # Удаляем оригинальные ступенчатые artist'ы
            for art in new_lines + new_colls:
                try:
                    art.remove()
                except Exception:
                    pass

            if not xs:
                continue

            xs = np.array(xs)
            ys = np.array(ys)
            sort_idx = np.argsort(xs)
            xs = xs[sort_idx]
            ys = ys[sort_idx]

            valid = ys < 1e20
            xs = xs[valid]
            ys = ys[valid]

            if len(xs) < 2:
                continue

            if len(xs) >= 4:
                x_new = np.linspace(xs[0], xs[-1], max(500, len(xs) * 5))
                try:
                    f = PchipInterpolator(xs, ys)
                    y_new = f(x_new)
                except Exception:
                    y_new = np.interp(x_new, xs, ys)
                ax.plot(x_new, y_new, color=color, lw=2.5, label=label)

            self.legend_handles.append(Line2D([0], [0], color=color, lw=1.5, label=label))

    def _load_line_coords(self) -> Optional[List[Tuple[float, float]]]:
        if self._line_coords is not None:
            return self._line_coords
        path = self.config.line_path
        if not path:
            print("Warning: Cross-section line_path is not set.")
            return None
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            print(f"Error reading cross-section line {path}: {e}")
            return None

        if self.global_crs:
            gdf = gdf.set_crs(self.global_crs, allow_override=True)

        if self.config.line_filter:
            gdf = self._apply_filter(gdf, self.config.line_filter)

        if gdf.empty:
            return None

        geom = gdf.geometry.iloc[0]
        if geom is None or geom.is_empty:
            return None

        if geom.geom_type == 'LineString':
            self._line_coords = list(geom.coords)
            # self._line_coords = list(geom.coords)[::-1]
            return self._line_coords

        if geom.geom_type == 'MultiLineString':
            longest = max(geom.geoms, key=lambda g: g.length, default=None)
            self._line_coords = list(longest.coords) if longest else None
            return self._line_coords

        print("Warning: Cross-section line geometry must be LineString.")
        return None

    def _label_section_ends(self, ax: plt.Axes) -> None:
        label_start = getattr(self.config, "line_label_start", "A")
        label_end = getattr(self.config, "line_label_end", "B")
        x_min, x_max = ax.get_xlim()
        y_pos = 1.02
        ax.text(
            x_min, y_pos, label_start,
            transform=ax.get_xaxis_transform(),
            ha='left', va='bottom'
        )
        ax.text(
            x_max, y_pos, label_end,
            transform=ax.get_xaxis_transform(),
            ha='right', va='bottom'
        )

    def _plot_surface_raster(self, ax: plt.Axes) -> None:
        path = getattr(self.config, "surface_raster_path", None)
        if not path:
            return
        line = self._load_line_coords()
        if not line:
            return
        step = float(getattr(self.config, "surface_sample_step", 50.0))
        if step <= 0:
            step = 50.0
        pts, dists = self._densify_line_with_distances(line, step)
        if not pts:
            return
        try:
            with rasterio.open(path) as src:
                sample_pts = pts
                if self.global_crs and src.crs and str(src.crs) != str(self.global_crs):
                    try:
                        transformer = Transformer.from_crs(
                            CRS.from_user_input(self.global_crs),
                            CRS.from_user_input(src.crs),
                            always_xy=True
                        )
                        xs, ys = zip(*pts)
                        xs_t, ys_t = transformer.transform(xs, ys)
                        sample_pts = list(zip(xs_t, ys_t))
                    except Exception as e:
                        print(f"Warning: Surface raster CRS differs; reprojection failed: {e}")
                vals = list(src.sample(sample_pts))
                vals = np.array(vals, dtype=float).reshape(-1)
                if src.nodata is not None:
                    vals = np.where(np.isclose(vals, src.nodata), np.nan, vals)
        except Exception as e:
            print(f"Error reading surface raster {path}: {e}")
            return

        mask = np.isfinite(vals)
        if not np.any(mask):
            return

        ax.plot(
            np.array(dists)[mask],
            vals[mask],
            color=getattr(self.config, "surface_color", "black"),
            linewidth=getattr(self.config, "surface_linewidth", 1.0),
            label=getattr(self.config, "surface_label", None),
            zorder=getattr(self.config, "surface_zorder", 5)
        )

    def _apply_section_ylim(self, ax: plt.Axes) -> None:
        if not getattr(self.config, "section_autolimit", True):
            return
        y_min, y_max = self._get_model_elevation_limits()
        if y_min is None or y_max is None:
            ax.relim()
            ax.autoscale_view()
            y_min, y_max = ax.dataLim.intervaly
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return
        if y_min == y_max:
            return
        pad_frac = getattr(self.config, "section_ylim_padding", 0.02)
        pad = (y_max - y_min) * pad_frac
        ax.set_ylim(y_min - pad, y_max + pad)

    def _get_model_elevation_limits(self):
        if self.model is None or not hasattr(self.model, "modelgrid"):
            return None, None
        grid = self.model.modelgrid
        try:
            top = np.asarray(grid.top, dtype=float)
            botm = np.asarray(grid.botm, dtype=float)
        except Exception:
            return None, None
        top_vals = top[np.isfinite(top)]
        botm_vals = botm[np.isfinite(botm)]
        if top_vals.size == 0 or botm_vals.size == 0:
            return None, None
        return float(np.min(botm_vals)), float(np.max(top_vals))

    @staticmethod
    def _densify_line_with_distances(line: List[Tuple[float, float]], step: float):
        if len(line) < 2:
            return [], []
        points = []
        dists = []
        total = 0.0
        for i in range(len(line) - 1):
            x0, y0 = line[i]
            x1, y1 = line[i + 1]
            seg_len = float(np.hypot(x1 - x0, y1 - y0))
            if seg_len == 0:
                continue
            n = max(1, int(seg_len / step))
            for j in range(n):
                t = j / n
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                if points and x == points[-1][0] and y == points[-1][1]:
                    continue
                points.append((x, y))
                dists.append(total + seg_len * t)
            total += seg_len
        points.append(line[-1])
        dists.append(total)
        return points, dists

    def _get_model_elevation_limits(self):
        if self.model is None or not hasattr(self.model, "modelgrid"):
            return None, None
        grid = self.model.modelgrid
        try:
            top = np.asarray(grid.top, dtype=float)
            botm = np.asarray(grid.botm, dtype=float)
        except Exception:
            return None, None

        top_vals = top[np.isfinite(top)]
        botm_vals = botm[np.isfinite(botm)]
        if top_vals.size == 0 or botm_vals.size == 0:
            return None, None
        return float(np.min(botm_vals)), float(np.max(top_vals))

    def draw_line_on_map(self, ax: plt.Axes) -> None:
        if not getattr(self.config, "show_line_on_map", False):
            return
        line = self._load_line_coords()
        if not line:
            return
        xs, ys = zip(*line)
        ax.plot(
            xs,
            ys,
            color=self.config.line_color,
            linewidth=self.config.line_width,
            linestyle='-',
            zorder=100
        )
        label_start = getattr(self.config, "line_label_start", "A")
        label_end = getattr(self.config, "line_label_end", "B")
        offset = getattr(self.config, "line_label_offset_points", 6)
        ax.annotate(
            label_start,
            xy=(xs[0], ys[0]),
            xytext=(offset, -offset),
            textcoords="offset points",
            fontsize=14,
            ha='left',
            va='bottom'
        )
        ax.annotate(
            label_end,
            xy=(xs[-1], ys[-1]),
            xytext=(offset, offset),
            textcoords="offset points",
            fontsize=14,
            ha='left',
            va='bottom'
        )

    @staticmethod
    def _apply_filter(gdf: gpd.GeoDataFrame, query: str) -> gpd.GeoDataFrame:
        try:
            col, val = query.split('=', 1)
            if col not in gdf.columns:
                print(f"Warning: Filter column '{col}' not found.")
                return gdf

            if gdf[col].dtype == 'O':
                return gdf[gdf[col] == val]
            try:
                num_val = float(val)
                return gdf[gdf[col] == num_val]
            except ValueError:
                return gdf[gdf[col] == val]
        except ValueError:
            return gdf

    def _calculate_auto_levels(self, data, n_levels=10):
        valid_data = data.compressed() if isinstance(data, np.ma.MaskedArray) else data[~np.isnan(data)]
        if valid_data.size == 0:
            return None

        vmin, vmax = valid_data.min(), valid_data.max()
        if vmin == vmax:
            return [vmin]

        raw_step = (vmax - vmin) / n_levels
        step = 10 ** np.floor(np.log10(raw_step))
        if raw_step / step > 5:
            step *= 5
        elif raw_step / step > 2:
            step *= 2

        levels = np.arange(np.floor(vmin / step) * step,
                           np.ceil(vmax / step) * step + step,
                           step)
        return levels

    def _load_model(self):
        ws = self.config.model_ws
        name = self.config.model_nam
        if not ws:
            print("Warning: Cross-section model_ws is not set.")
            return None
        cache_key = (ws, name)

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        print(f"Loading MODFLOW model from {ws}...")
        try:
            sim = flopy.mf6.MFSimulation.load(
                sim_ws=ws,
                verbosity_level=0
            )
            # model = sim.get_model(name) if name else sim.get_model()
            model = sim.get_model()
            self._model_cache[cache_key] = model
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _get_array_data(self):
        param_name = self.config.parameter.lower()
        if param_name == 'head':
            return self._get_heads_data()
        if param_name == 'k1':
            return self._get_npf_array('k')
        if param_name == 'k2':
            return self._get_npf_array('k22')
        if param_name == 'k3':
            return self._get_npf_array('k33')
        return None

    def _get_npf_array(self, attr_name: str):
        if not hasattr(self.model, 'npf'):
            return None
        npf = self.model.npf
        if hasattr(npf, attr_name):
            arr = getattr(npf, attr_name)
            return arr.array if hasattr(arr, 'array') else arr
        if hasattr(npf, 'k'):
            arr = npf.k
            return arr.array if hasattr(arr, 'array') else arr
        return None

    def _get_heads_data(self):
        model_name = self.config.model_nam or getattr(self.model, 'name', None)
        if not model_name:
            return None
        head_file = os.path.join(self.config.model_ws, f"{model_name}.hds")
        try:
            head = flopy.utils.HeadFile(head_file)
            return head.get_data(kstpkper=(0, self.config.stress_period))
        except Exception as e:
            print(f"Error reading heads: {e}")
        try:
            if hasattr(self.model, 'output') and hasattr(self.model.output, 'head'):
                head_obj = self.model.output.head()
                if head_obj is not None:
                    return head_obj.get_data(kstpkper=(0, self.config.stress_period))
        except Exception:
            return None


class FlopyLayer(IMapLayer):
    _model_cache = {}

    def __init__(self, config: FlopyLayerConfig, global_crs: Optional[str] = None):
        super().__init__(config, global_crs)
        self.config: FlopyLayerConfig = config
        self.model = self._load_model()

    def draw(self, ax: plt.Axes) -> None:
        """Главный метод-оркестратор."""
        try:
            if not self.model:
                return
            pmv = flopy.plot.PlotMapView(model=self.model, layer=self.config.layer, ax=ax)

            if self.config.parameter:
                self._draw_parameter(pmv)

            if self.config.grid_enabled:
                self._draw_grid(pmv)

            if self.config.bc_enabled:
                self._draw_boundary_conditions(pmv)

        except Exception as e:
            print(f"Failed to draw Flopy layer: {e}")

    def _calculate_auto_levels(self, data, n_levels=10):
        """Создает массив 'круглых' уровней для изолиний."""
        # Берем только реальные значения для расчета диапазона
        valid_data = data.compressed() if isinstance(data, np.ma.MaskedArray) else data[~np.isnan(data)]
        valid_data = valid_data[valid_data > -1e10]  # Исключаем dry cells

        if valid_data.size == 0:
            return None

        vmin, vmax = valid_data.min(), valid_data.max()
        if vmin == vmax:
            return [vmin]

        # Логика подбора красивого шага (1, 2, 5, 10...)
        raw_step = (vmax - vmin) / n_levels
        step = 10 ** np.floor(np.log10(raw_step))
        if raw_step / step > 5:
            step *= 5
        elif raw_step / step > 2:
            step *= 2

        levels = np.arange(np.floor(vmin / step) * step,
                           np.ceil(vmax / step) * step + step,
                           step)
        return levels

    def _draw_contours(self, pmv: flopy.plot.PlotMapView, data):
        """Отрисовка изолиний с жесткой обрезкой по idomain."""
        if data is None:
            return

        grid = self.model.modelgrid
        inactive_mask = grid.idomain[self.config.layer] <= 0

        data_masked = np.ma.masked_where(inactive_mask | (data < -1e10), data)

        c_style = self.config.contour_style
        levels = c_style.levels
        if not levels:
            levels = self._calculate_auto_levels(data_masked)

        cs = pmv.contour_array(
            data_masked,
            levels=levels,
            colors=c_style.colors,
            linewidths=c_style.linewidths,
            zorder=self.config.zorder + 0.3
        )

        if cs is not None:
            ax = pmv.ax
            ax.clabel(cs, inline=True,
                      fontsize=c_style.fontsize,
                      fmt='%1.1f')

    def _load_model(self):
        """Загружает или достает из кэша объект модели MF6."""
        ws = self.config.model_ws
        name = self.config.model_nam
        cache_key = (ws, name)

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        print(f"Loading MODFLOW model from {ws}...")
        try:
            sim = flopy.mf6.MFSimulation.load(
                sim_ws=ws,
                # load_only=['dis', 'disv', 'npf', 'rch', 'ic', 'riv', 'drn', 'wel', 'chd', 'ghb'],
                verbosity_level=0
            )
            model = sim.get_model()
            self._model_cache[cache_key] = model
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _parameter_norm(self, data):
        style = self.config.style
        valid_data = data[~np.isnan(data)]
        for val in self.config.masked_values:
            valid_data = valid_data[valid_data != val]

        if valid_data.size == 0:
            return Normalize()

        if style.classification == 'quantile':
            bins = np.percentile(valid_data, np.linspace(0, 100, style.n_classes + 1))
            bins = np.unique(bins)

            if len(bins) < 3:
                print("Warning: Too many duplicate values for quantile classification. Falling back to linear.")
                return Normalize(vmin=valid_data.min(), vmax=valid_data.max())
            norm = BoundaryNorm(boundaries=bins, ncolors=len(bins) - 1)
            return norm
        else:
            data_masked = np.ma.masked_invalid(data)
            for val in self.config.masked_values:
                data_masked = np.ma.masked_equal(data_masked, val)

            if self.config.log_scale:
                data_masked = np.ma.masked_less_equal(data_masked, 0)

            if data_masked.count() == 0:
                print(f"Warning: All data masked for {self.config.parameter}. Skipping plot.")
                return

            vmin = style.vmin if style.vmin is not None else data_masked.min()
            vmax = style.vmax if style.vmax is not None else data_masked.max()

            if vmin == vmax:
                vmin = vmin - 0.1 * abs(vmin) if vmin != 0 else -0.1
                vmax = vmax + 0.1 * abs(vmax) if vmax != 0 else 0.1

            if self.config.log_scale:
                if vmin <= 0: vmin = 1e-10
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

        return norm

    def _draw_parameter(self, pmv: flopy.plot.PlotMapView):
        data = self._get_array_data()
        print(data)
        if data is None or np.all(np.isnan(data)):
            print(f"Warning: No data to plot for {self.config.parameter}")
            return

        style = self.config.style
        norm = self._parameter_norm(data)
        cmap_name = style.cmap or 'viridis'

        # Определяем количество цветов для палитры
        if isinstance(norm, BoundaryNorm):
            # Количество цветов в BoundaryNorm ВСЕГДА на 1 меньше, чем количество границ
            n_colors = len(norm.boundaries) - 1
            cmap = plt.get_cmap(cmap_name, n_colors)
        else:
            cmap = plt.get_cmap(cmap_name)
        self.mappable = pmv.plot_array(
            data,
            masked_values=self.config.masked_values,
            cmap=cmap,
            norm=norm,
            alpha=style.alpha,
            zorder=self.config.zorder
        )

        if self.config.contours and data is not None:
            self._draw_contours(pmv, data)

    def _draw_grid(self, pmv: flopy.plot.PlotMapView):
        """Отрисовка сетки."""
        pmv.plot_grid(
            color=self.config.grid_color,
            linewidth=self.config.grid_linewidth,
            zorder=self.config.zorder + 0.2
        )

    def _draw_boundary_conditions(self, pmv: flopy.plot.PlotMapView):
        """Отрисовка пакетов граничных условий."""
        for bc_type in self.config.bc_enabled:
            color = self.config.bc_colors.get(bc_type, 'blue')
            try:
                pmv.plot_bc(
                    name=bc_type,
                    color=color,
                    kper=self.config.stress_period,
                    zorder=self.config.zorder + 0.1
                )
            except Exception:
                print(f"Warning: BC package '{bc_type}' not found or failed to plot.")

    def _get_array_data(self):
        """Верхнеуровневый метод извлечения данных."""
        param_name = self.config.parameter.lower()
        print(param_name)
        if ':' in param_name:
            return self._extract_package_data(param_name)

        return self._extract_property_array(param_name)

    def _extract_package_data(self, param_name: str):
        """Извлекает данные из списочных пакетов (DRN, RIV, WEL и т.д.)."""
        pkg_name, var_name = param_name.split(':')
        stress_period = self.config.stress_period
        layer_idx = self.config.layer

        pkg = self._find_package(pkg_name)
        if not (pkg and hasattr(pkg, 'stress_period_data')):
            return None

        sp_data = pkg.stress_period_data.get_data(key=stress_period)
        if sp_data is None or len(sp_data) == 0:
            return None

        target_field = next((n for n in sp_data.dtype.names if var_name in n.lower()), None)
        if not target_field:
            return None

        return self._map_list_to_array(sp_data, target_field, layer_idx)

    def _find_package(self, pkg_name: str):
        """Ищет пакет по имени или частичному совпадению типа."""
        pkg = self.model.get_package(pkg_name)
        if pkg:
            return pkg

        for p in self.model.get_package_list():
            if pkg_name in p.lower():
                return self.model.get_package(p)
        return None

    def _map_list_to_array(self, sp_data, field, layer_idx):
        """Переносит данные из stress_period_data (list-based) в массив (grid-based)."""
        grid = self.model.modelgrid

        if grid.grid_type == 'structured':
            arr = np.full((grid.nrow, grid.ncol), np.nan)
        else:
            arr = np.full((grid.ncpl,), np.nan)

        cellids = sp_data['cellid']
        values = sp_data[field]

        for i, cid in enumerate(cellids):
            if cid[0] == layer_idx:
                if grid.grid_type == 'structured':
                    arr[cid[1], cid[2]] = values[i]
                else:  # vertex
                    arr[cid[1]] = values[i]
        return arr

    def _extract_property_array(self, param_name: str):
        """Извлекает данные из полных массивов (K, Recharge и т.д.)."""
        layer_idx = self.config.layer
        stress_period = self.config.stress_period

        if param_name in ['k', 'hk', 'k11'] and hasattr(self.model, 'npf'):
            return self.model.npf.k.array[layer_idx]
        if param_name in ['k2', 'hk22', 'k22'] and hasattr(self.model, 'npf'):
            return self.model.npf.k22.array[layer_idx]
        if param_name in ['rch', 'recharge'] and hasattr(self.model, 'rch'):
            return self._get_recharge_data(stress_period)
        if param_name in ['ghb'] and hasattr(self.model, 'ghb'):
            return self._get_ghb_data(stress_period, layer_idx)
        if param_name in ['drn'] and hasattr(self.model, 'drn'):
            return self._get_drn_data(stress_period, layer_idx)
        if param_name == "heads":
            return self._get_heads_data()

        return None

    def _get_heads_data(self):
        head_file = os.path.join(self.config.model_ws, f"{self.config.model_nam}.hds")
        head = flopy.utils.HeadFile(head_file)
        times = head.get_times()
        hds = head.get_data(totim=times[self.config.stress_period])
        if hds.ndim == 2:
            head_vals = hds[0, :]
        else:
            head_vals = hds
        # === Присоединяем к GeoDataFrame ===
        # return head.get_alldata()[self.config.stress_period][self.config.layer][-1]
        return head_vals[self.config.layer][0]


    def _get_recharge_data(self, stress_period):
        """Вспомогательный метод специально для сложной структуры RCH."""
        rch_array = self.model.rch.recharge.array
        # Упрощаем логику размерностей: берем последний и предпоследний индексы
        if rch_array.ndim == 4:  # (sp, lay, row, col)
            return rch_array[stress_period, 0, :, :]
        if rch_array.ndim == 3:  # (sp, lay, ncpl)
            return rch_array[stress_period, 0, :]
        return rch_array[stress_period]

    def _get_ghb_data(self, stress_period, layer_idx):
        """Вспомогательный метод специально для сложной структуры RCH."""
        array = self.model.ghb.stress_period_data.to_array()['bhead']
        return array[layer_idx]

    def _get_drn_data(self, stress_period, layer_idx):
        """Вспомогательный метод специально для сложной структуры RCH."""
        array = self.model.drn.stress_period_data.to_array()['elev']
        return array[layer_idx]

    def get_legend_handles(self) -> List[Any]:
        handles = []
        if self.config.bc_enabled:
            for bc in self.config.bc_enabled:
                color = self.config.bc_colors.get(bc, 'blue')
                handles.append(Patch(facecolor=color, label=self.config.label if self.config.label else bc.upper()))
        return handles


class PestLayer(FlopyLayer):
    # TODO: Нужно сделать базовый класс с общими методами для PestLayer, FlopyLayer
    def __init__(self, config: PestLayerConfig, global_crs: Optional[str] = None):
        super().__init__(config, global_crs)
        self.config: PestLayerConfig = config
        self.model = self._load_model()

    def get_tpl_map(self, tpl_path):
        with open(tpl_path, 'r') as f:
            header = f.readline()
            marker = header.strip().split()[1]

            mapping = []
            for line in f:
                if marker in line:
                    parts = line.split()
                    pp_name = parts[0]
                    pest_name = line.split(marker)[1].strip()
                    mapping.append({"pp_name": pp_name, "pest_name": pest_name})
        return pd.DataFrame(mapping)

    def get_vgrid_std(self):
        ws = self.config.model_ws
        mapping_df = self.get_tpl_map(os.path.join(ws, self.config.tpl_file))
        pe = pyemu.ParameterEnsemble.from_csv(pst=pyemu.Pst(os.path.join(ws, self.config.pst)),
                                              filename=os.path.join(ws, self.config.ensemble_file))
        param_std = pe.std(axis=0).to_frame(name="std")

        merged_map = mapping_df.merge(param_std, left_on="pest_name", right_index=True)

        pp_df = pyemu.pp_utils.pp_tpl_to_dataframe(os.path.join(ws, self.config.tpl_file))
        pp_df = pp_df.merge(merged_map[['pp_name', 'std']], left_on="name", right_on="pp_name")

        pp_df.loc[:, "parval1"] = pp_df["std"]

        return pyemu.geostats.fac2real(pp_df,
                                       factors_file=os.path.join(ws, self.config.fac_file),
                                       out_file=None
                                       ).flatten()

    def _extract_property_array(self, param_name: str):
        """Извлекает данные из полных массивов (K, Recharge и т.д.)."""

        if param_name == "std_npf":
            return self.get_vgrid_std()

        return None
