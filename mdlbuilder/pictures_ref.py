import os
from textwrap import wrap
import math

import flopy
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
import rasterio
from rasterio.plot import show, adjust_band
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.mask import mask
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from textwrap import wrap
from adjustText import adjust_text


class OutputPicture:
    def __init__(self, epsg, model, name, subspace, base_objects, config, extent=None):
        rc('text', usetex=False)
        self.epsg = epsg
        self.model = model
        self.name_pic = name
        self.output_path = self._create_dir(f"output/pictures/{subspace}")
        self.subplot = config.get("subplot")
        self.scale = config.get("scale")
        if self.subplot:
            self.fig = plt.figure()
            RESIZE_FACTOR = 2.5
            self.fig.set_size_inches(self.fig.get_size_inches() * RESIZE_FACTOR)
            self.ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
            self.ax_inset = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
            self.ax_legend = plt.subplot2grid((3, 3), (2, 2))
            self.ax_legend.set_axis_off()
            self.fig.subplots_adjust(hspace=0.1, wspace=0.1)
            self.disable_ticks(self.ax_inset)
            self.set_sizebar(self.ax_inset, self.scale / 50)
        else:
            self.fig, self.ax = plt.subplots(1, 1)
            self.fig.set_size_inches(14, 16)
        # self.ax.set_axis_off()

        self.disable_ticks(self.ax)
        if self.scale:
            self.set_sizebar(self.ax, self.scale)

        self.handles, self.labels = [], []
        self.base_objects = base_objects
        self.base_layers = config.get("baselayers")
        self.basemaps = config.get("basemap")
        self.zoom = config.get("zoom")
        self.map = config.get("map")
        self.contours = config.get("contours")
        self.clr_points = config.get("clr_points")
        if extent:
            self.border = self.border_poly(extent)
        else:
            self.border = None
        self.mapview = self.create_mapview()

    @staticmethod
    def _create_dir(path: str) -> str:
        out_dir = os.path.join(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir

    def disable_ticks(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def set_sizebar(self, ax, scale=500):
        ax.add_artist(AnchoredSizeBar(ax.transData,
                                           scale, f'{scale} м', 'lower center',
                                           pad=0.1,
                                           color='black',
                                           frameon=False,
                                           size_vertical=1))
    def add_basemap(self, ax):
        self.zoomed(ax, self.zoom)
        cx.add_basemap(ax, crs=self.epsg, attribution='', alpha=0.4, source=cx.providers.OpenStreetMap.Mapnik)

    def get_baselayer(self, blay_path):
        blay_gdf = gpd.read_file(blay_path, encoding="cp1251")
        blay_gdf.crs = self.epsg
        if self.border is not None:
            blay_gdf = gpd.sjoin(blay_gdf, self.border, how="inner")
            if blay_gdf.empty:
                blay_gdf = gpd.read_file(blay_path)
        return blay_gdf

    def base_handle_labels(self):
        title = r"$\boldsymbol{Условные\ обозначения}$"
        title_proxy = Line2D([], [], color="none", label=title)
        self.handles = [title_proxy]
        self.labels = [title]
        # if self.plot_grid_ins:
        #     lines = [
        #         Line2D([0], [0], linestyle="none", linewidth=0.2, marker=r"$▦$", markersize=20, color=t.get_colors())
        #         for t in self.ax.collections[:] if type(t) is LineCollection
        #     ]
        #     labels = [t.get_label() for t in self.ax.collections[:]]
        #     line_legend_handles = [lines[0]]
        #     line_legend_labels = [labels[0]]
        #     self.handles += line_legend_handles
        #     self.labels += line_legend_labels

    def other_handles(self, typo="line", label="", options=None):
        marker_options = {"marker": "s", "linewidth": 0.6, "markersize": 15,
                          "color": "none", "markeredgewidth": 1, "markeredgecolor": "green"}
        if label:
            for k, v in options.items():
                marker_options[k] = v
            if typo == "line":
                self.handles += [Line2D([0], [0], **marker_options)]
                self.labels += [label]
            elif typo == "patch":
                self.handles += [mpatches.Patch(label=label, **marker_options)]
                self.labels += [label]

    def wrap_legend(self):
        self.labels = [self.labels[0]] + ['\n'.join(wrap(l, 30)) for l in self.labels[1:]]

    def create_legend(self):
        if self.subplot:
            leg = self.ax_legend.legend(handles=self.handles, labels=self.labels, prop={'size': 14},
                                        loc='center', bbox_to_anchor=(0.5, 1))
        else:
            leg = self.ax.legend(handles=self.handles, labels=self.labels, loc='upper left', bbox_to_anchor=(1.04, 1),
                                 frameon=False,
                                 prop={'size': 14})

    def zoomed(self, ax, zoom=10):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lonpad, latpad = zoom * (xmax - xmin), zoom * (ymax - ymin)
        ax.axis("scaled")
        ax.axis([xmin - lonpad, xmax + lonpad, ymin - latpad, ymax + latpad])

    def create_mapview(self):
        return flopy.plot.PlotMapView(model=self.model, layer=0)

    def plot_grid(self):
        self.mapview.plot_grid(label="Модельная сетка", colors='black', linewidth=0.3, facecolor="white", alpha=0.4)

    def get_heads(self):
        head_file = os.path.join(self.model.model_ws, f"{self.model.name}.hds")
        head = flopy.utils.HeadFile(head_file)
        # return head.get_data(kstpkper=self.kper if self.kper else (0, 0))
        return head.get_data(kstpkper=(0, 0))

    def get_npf(self):
        return self.model.npf.k.array

    def get_rch(self):
        return self.model.rch.recharge.array

    def get_top(self):
        return self.model.dis.top.array

    def get_bot(self):
        return self.model.dis.botm.array

    def get_lay(self):
        return self.map.get("layer") if self.map.get("layer") is not None else 1

    def plot_raster(self):
        with rasterio.open(self.map.get("data"), crs=self.epsg) as src:

            if self.border is not None:
                out_image, out_transform = self.crop_raster(src)
            else:
                out_image, out_transform = src.read(1), src.transform

            if self.map.get("crop"):

                cropb = gpd.read_file(self.map.get("crop")).dissolve()
                cropb = cropb.to_crs(src.crs)
                cropb = gpd.clip(cropb, self.border, keep_geom_type=True)
                out_image, out_transform = mask(src, cropb.geometry, crop=True)
                out_image = np.ma.masked_array(out_image, mask=(out_image == src.nodata))

            if self.map.get("options"):
                pltview = show(out_image, ax=self.ax, transform=out_transform,
                     **self.map.get("options"))
            else:
                pltview = show(out_image, ax=self.ax, transform=out_transform)
        if self.map.get("label"):
            im = pltview.get_images()[0]
            cbar = plt.colorbar(im, shrink=0.75, orientation="horizontal", pad=0.01)
            if self.map.get("label"):
                cbar.ax.set_xlabel(f'{self.map.get("label")}')

    def form_data(self):
        data_name = self.map.get("data")
        lay = self.get_lay() - 1
        data_match = {"heads": self.get_heads()[lay], "hk": np.log10(self.get_npf()[lay]),
                      "rch": self.get_rch(), "top": self.get_top(), "bot": self.get_bot()[lay]}
        data = data_match.get(data_name)
        if lay > 0:
            idomain = np.where(self.model.modelgrid.botm[lay - 1] - self.model.modelgrid.botm[lay] <= 0.11, 0, 1)
        else:
            idomain = np.where(self.model.modelgrid.top - self.model.modelgrid.botm[0] <= 0.11, 0, 1)
        if data_name == "hk":
            for i, idm in enumerate(idomain):
                if idm == 0:
                    data[i] = np.nan
        return data

    def plot_array(self):
        data = self.form_data()
        if data is not None:

            pltview = self.mapview.plot_array(a=data, alpha=0.7, cmap=self.map.get("cmap", "Blues"))
            self.mapview.plot_ibound()
            cbar = plt.colorbar(pltview, shrink=0.75, orientation="horizontal", pad=0.01)
            if self.map.get("label"):
                cbar.ax.set_xlabel(f'{self.map.get("label")}')

    def border_poly(self, extent):
        border = gpd.read_file(extent)
        border.crs = self.epsg
        return border

    def crop_maps(self, src):
        out_image, out_transform = mask(src, self.border.geometry, crop=True)
        out_image = np.ma.masked_array(out_image, mask=(out_image == src.nodata))
        return out_image, out_transform

    def crop_raster(self, src):
        border = self.border.to_crs(src.crs)
        out_image, out_transform = mask(src, border.geometry, crop=True)
        out_image = np.ma.masked_array(out_image, mask=(out_image == src.nodata))
        return out_image, out_transform

    def init_cnt_levels(self, img):
        if np.nanmean(img) > 1:
            vmin = round(np.nanmin(img) / 5, 0) * 5
            vmax = round(np.nanmax(img) / 5, 0) * 5
        else:
            vmin = -1
            vmax = 1

        if self.contours.get("interval"):
            levels = np.arange(vmin, vmax, self.contours.get("interval"))
        else:
            levels = np.linspace(vmin, vmax, 10)
        return levels

    def plot_cnt(self):
        with rasterio.open(self.contours.get("path"), crs=self.epsg) as src:
            if self.border is not None:
                out_image, out_transform = self.crop_raster(src)
            else:
                out_image, out_transform = src.read(1), src.transform

            if self.contours.get("crop"):

                cropb = gpd.read_file(self.contours.get("crop")).dissolve()
                cropb = cropb.to_crs(src.crs)
                cropb = gpd.clip(cropb, self.border, keep_geom_type=True)
                out_image, out_transform = mask(src, cropb.geometry, crop=True)
                out_image = np.ma.masked_array(out_image, mask=(out_image == src.nodata))
            levels = self.init_cnt_levels(out_image)
            if self.contours.get("options"):
                show(out_image, ax=self.ax, transform=out_transform, contour=True, levels=levels,
                     **self.contours.get("options"))
            else:
                show(out_image, ax=self.ax, transform=out_transform, contour=True, levels=levels)
        if self.contours.get("label"):
            self.cnt_handles(self.contours.get("options"), self.contours.get("label"))

    def cnt_handles(self, cnt_options, label):
        marker_options = {"marker": "", "linewidth": 1, "linestyle": '-', 'color': cnt_options.get("colors")}
        self.handles += [Line2D([0], [0], **marker_options)]
        self.labels += [label]

    def set_annotations(self, ax, gdf, anno):
        texts = []
        # trans_offset = mtransforms.offset_copy(self.ax.transData, fig=self.fig,
        #                                        x=0.01, y=0.01, units='inches')
        for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf[anno.get("by")]):
            texts.append(ax.text(x, y, label, fontsize=anno.get("fontsize", 4), ha='center', va='top'))
                                      # bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3')))
        adjust_text(texts, ax=ax,
                    only_move={'text': 'xy', 'points': 'xy'},
                    # expand_points=(1.2, 1.4),  # Expands the distance points can be moved
                    # expand_text=(1.2, 1.4),  # Expands the distance text can be moved
                    # force_text=0.5,  # Force of repulsion between texts
                    # force_points=0.5,  # Force of repulsion from points
                    autoalign='xy',  # Automatically align text for better fit
                    # lim=150
                    )#, expand=(2, 3)

    def res_legend(self, ax_res):
        legend = ax_res.get_legend()
        for ea in legend.legend_handles:
            ea.set_markeredgecolor('black')
            ea.set_markeredgewidth(1)
        res_handles, res_labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
        self.ax.get_legend().remove()
        title_res = "Невязки (факт. - мод.), м"
        title_res_proxy = Line2D([], [], color="none", label=title_res)
        self.handles += [title_res_proxy] + res_handles
        self.labels += [title_res] + res_labels

    def plot_clr_points(self):
        cmap_res = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
        clr_gdf = self.get_baselayer(self.clr_points.get("path"))
        clr_gdf = clr_gdf.dropna(subset=self.clr_points.get("by"))
        print(clr_gdf)
        print(clr_gdf.columns)
        ax_res = clr_gdf.plot(ax=self.ax, column=self.clr_points.get("by"), legend=True, cmap=cmap_res, scheme='quantiles', k=10,
                              edgecolors='black')
        self.res_legend(ax_res)

    def get_buffer_extent(self):
        if self.border is not None:
            return (self.border.total_bounds[2] - self.border.total_bounds[0]) / 10

    def show_map(self):
        datasets = [rasterio.open(f.get("path")) for f in self.basemaps]
        out_image, out_transform = merge(datasets)
        out_meta = datasets[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(out_image)
                if self.border is not None:
                    buf = self.get_buffer_extent()
                    out_image, out_transform = mask(
                        dataset,
                        shapes=[shapely.geometry.box(*self.border.total_bounds).buffer(buf)],
                        crop=True
                    )
                    out_image = np.ma.masked_array(out_image, mask=(out_image == dataset.nodata))
                show(adjust_band(out_image[:3]), ax=self.ax, transform=out_transform)

    def set_ax_border(self, ax):
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1)

    def plot(self):
        self.base_handle_labels()
        if self.base_objects:
            gdf = self.get_baselayer(self.base_objects.get("path"))
            gdf.plot(ax=self.ax, **self.base_objects.get("options"))
            self.other_handles(typo=self.base_objects.get("type_legend"),
                               label=self.base_objects.get("label"),
                               options=self.base_objects.get("marker_options"))
        if self.map:
            if isinstance(self.map.get("data"), str) and os.path.isfile(self.map.get("data")):
                self.plot_raster()
            else:
                self.plot_array()
        if self.contours:
            self.plot_cnt()
        if self.base_layers:
            for blay_inf in self.base_layers:
                gdf = self.get_baselayer(blay_inf.get("path"))
                if blay_inf.get("options"):
                    gdfplot = gdf.plot(ax=self.ax, **blay_inf.get("options"))
                    annotations = blay_inf.get("annotations")
                    if annotations:
                        self.set_annotations(self.ax, gdf, annotations)
                    self.other_handles(typo=blay_inf.get("type_legend"),
                                       label=blay_inf.get("label"),
                                       options=blay_inf.get("marker_options"))
                else:
                    gdf.plot(ax=self.ax)
        if self.subplot:
            gdf = self.get_baselayer(self.base_objects.get("path"))
            gdf.plot(ax=self.ax_inset, **self.base_objects.get("options"))
            for blay_inf in self.subplot:
                gdf = self.get_baselayer(blay_inf.get("path"))
                if blay_inf.get("options"):
                    gdfplot = gdf.plot(ax=self.ax_inset, **blay_inf.get("options"))
                    annotations = blay_inf.get("annotations")
                    if annotations:
                        self.set_annotations(self.ax_inset, gdf, annotations)
                else:
                    gdf.plot(ax=self.ax_inset)
                self.other_handles(typo=blay_inf.get("type_legend"),
                                   label=blay_inf.get("label"),
                                   options=blay_inf.get("marker_options"))
            self.add_basemap(self.ax_inset)
            self.set_ax_border(self.ax_inset)
        if self.basemaps:
            for bm in self.basemaps:
                self.show_map()

        self.add_basemap(self.ax)

        if self.clr_points:
            self.plot_clr_points()
        self.wrap_legend()
        self.create_legend()
        self.set_ax_border(self.ax)
        output = os.path.join(self.output_path, f"{self.name_pic}.png")
        plt.savefig(output, dpi=200, bbox_inches='tight')
