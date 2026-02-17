import os
from typing import Any, List, Optional, Dict

import numpy as np
import geopandas as gpd
import rasterio
import contextily as ctx
import flopy
from rasterio.mask import mask
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from mfbuilder.maps.protocols import IMapLayer
from mfbuilder.dto.maps import VectorLayerConfig, RasterLayerConfig, BasemapConfig, FlopyLayerConfig


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
                custom_text = 'EPSG:28412'
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

    def draw(self, ax: plt.Axes) -> None:
        path = self.config.path

        try:
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

            plot_kwargs = self._prepare_style(ax)
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

    def _prepare_style(self, ax: plt.Axes) -> Dict[str, Any]:
        style = self.config.style
        kwargs = {
            'ax': ax,
            'zorder': self.config.zorder,
            'label': self.config.label,
            'alpha': style.alpha
        }

        if style.color: kwargs['color'] = style.color
        if style.edgecolor: kwargs['edgecolor'] = style.edgecolor
        if style.linewidth: kwargs['linewidth'] = style.linewidth
        if style.markersize: kwargs['markersize'] = style.markersize
        if style.marker: kwargs['marker'] = style.marker
        if style.linestyle: kwargs['linestyle'] = style.linestyle
        if style.facecolor: kwargs['facecolor'] = style.facecolor
        if style.cmap: kwargs['cmap'] = style.cmap
        if 'color' not in kwargs and 'facecolor' not in kwargs and not style.cmap:
            kwargs['color'] = 'blue'

        return kwargs

    def get_legend_handles(self) -> List[Any]:
        """
        Генерирует элементы легенды на основе РЕАЛЬНОГО типа геометрии.
        """
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
                markersize=(style.markersize or 5) / 2,
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
            placements = [('left', 'bottom', 5, 5), ('right', 'bottom', -5, 5),
                          ('center', 'top', 0, -5), ('center', 'bottom', 0, 5)]

            for ha, va, off_x, off_y in placements:
                t_obj = ax.annotate(
                    label_text, xy=(p.x, p.y), xytext=(off_x, off_y),
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=lbl_conf.fontsize, fontweight=lbl_conf.fontweight,
                    color=lbl_conf.color, zorder=zorder, clip_on=True
                )

                bbox = t_obj.get_window_extent(renderer).expanded(1.05, 1.05)
                overlap = any(bbox.overlaps(o) for o in occupied_bboxes)

                if not overlap:
                    occupied_bboxes.append(bbox)
                    if lbl_conf.halo:
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

        if param_name in ['rch', 'recharge'] and hasattr(self.model, 'rch'):
            return self._get_recharge_data(stress_period)

        if param_name == "heads":
            return self._get_heads_data()

        return None

    def _get_heads_data(self):
        head_file = os.path.join(self.config.model_ws, f"{self.config.model_nam}.hds")
        head = flopy.utils.HeadFile(head_file)

        return head.get_alldata()[self.config.stress_period][self.config.layer][0]

    def _get_recharge_data(self, stress_period):
        """Вспомогательный метод специально для сложной структуры RCH."""
        rch_array = self.model.rch.recharge.array
        # Упрощаем логику размерностей: берем последний и предпоследний индексы
        if rch_array.ndim == 4:  # (sp, lay, row, col)
            return rch_array[stress_period, 0, :, :]
        if rch_array.ndim == 3:  # (sp, lay, ncpl)
            return rch_array[stress_period, 0, :]
        return rch_array[stress_period]

    def get_legend_handles(self) -> List[Any]:
        handles = []
        if self.config.bc_enabled:
            for bc in self.config.bc_enabled:
                color = self.config.bc_colors.get(bc, 'blue')
                handles.append(Patch(facecolor=color, label=self.config.label if self.config.label else bc.upper()))
        return handles