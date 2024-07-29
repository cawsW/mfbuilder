import os
import math
import flopy
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from textwrap import wrap

rc('text', usetex=False)
newpth = "../uncertainty_model"


class OutputPicture:
    @staticmethod
    def get_model():
        sim = flopy.mf6.MFSimulation.load(sim_ws=newpth, exe_name="mf6")
        return sim.get_model()

    @staticmethod
    def mean_slope(x, y):
        # print(points)
        # x, y = points.x, points.y
        slope, intercept = np.polyfit(x, y, 1)
        angle = math.atan(slope)
        return angle * (180 / math.pi)

    def __init__(self, epsg, output_path, scale, name_pic, arr_plot=False, layer=0,
                 residuals=False, base_layers=None, contours=None, plot_grid_ins=False, arr_label=None, kper=None):
        self.epsg = epsg
        self.model = self.get_model()
        self.name_pic = name_pic
        self.arr_label = arr_label
        self.base_object = base_layers.pop("base_object") if base_layers.get("base_object") else None
        self.output_path = output_path
        self.plot_grid_ins = plot_grid_ins
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.set_size_inches(14, 16)
        self.ax.set_axis_off()
        self.ax.add_artist(AnchoredSizeBar(self.ax.transData,
                                           scale, f'{scale} м', 'lower center',
                                           pad=0.1,
                                           color='black',
                                           frameon=False,
                                           size_vertical=1))
        self.mapview = self.create_mapview()
        self.arr_plot = arr_plot
        self.residuals = residuals
        self.layer = layer
        self.handles, self.labels = [], []
        if base_layers:
            self.base_layers = base_layers
        self.contours = contours
        self.kper = kper

    def add_basemap(self):
        cx.add_basemap(self.ax, crs=self.epsg, attribution='', alpha=0.4, source=cx.providers.OpenStreetMap.Mapnik)

    def get_baselayer(self, blay_path):
        blay_gdf = gpd.read_file(blay_path)
        blay_gdf.crs = self.epsg
        return blay_gdf

    def river_handles(self, label=None):
        self.handles += [
            Line2D([0], [0], linestyle="none", marker='$\sim$', linewidth=3.5, markersize=20, color='#4793AF')]
        self.labels += [label if label else "Реки"]

    def plot_river(self, gdf):
        gdf.plot(ax=self.ax, color='#4793AF', linewidth=3.5)
        points = gdf.geometry.get_coordinates()
        for i, row in gdf.iterrows():
            midpoint = row.geometry.interpolate(0.5, normalized=True)
            nextpoint = row.geometry.interpolate(0.3, normalized=True)
            angle = self.mean_slope([midpoint.x, nextpoint.x], [midpoint.y, nextpoint.y])
            if angle < 0:
                x_text = 70
            else:
                x_text = -200
            plt.text(nextpoint.x + x_text, nextpoint.y, ha='right', s=row["name"], va='bottom', fontdict=None,
                     transform_rotates_text=True, rotation=angle, rotation_mode='anchor', fontsize=8, color='#4793AF')

    def drn_handles(self):
        self.handles += [Line2D([0], [0], linestyle="none", marker='s', linewidth=0.2, markersize=20, color="none",
                                markeredgewidth=1, markeredgecolor="green")]
        self.labels += ["Карьер"]

    def other_handles(self, type="line", marker="s", label="", marker_options={"marker": "s", "linewidth": 0.2,
                                                                               "markersize": 20, "color": "none",
                                                                               "markeredgewidth": 1,
                                                                               "markeredgecolor": "green"}):
        if type == "line":
            self.handles += [Line2D([0], [0], **marker_options)]
            self.labels += [label]
        elif type == "patch":
            # mpatches.Patch(facecolor='w', hatch='\\\\\\\\', edgecolor='k', label=label)
            self.handles += [mpatches.Patch(label=label, **marker_options)]
            self.labels += [label]

    def plot_drn(self, gdf):
        gdf.plot(ax=self.ax, linewidth=2, edgecolor="green", color="none")
        points = gdf.geometry.get_coordinates()

    def base_handle_labels(self):
        title = r"$\boldsymbol{Условные\ обозначения}$"
        title_proxy = Line2D([], [], color="none", label=title)
        self.handles = [title_proxy]
        self.labels = [title]
        if self.plot_grid_ins:
            lines = [
                Line2D([0], [0], linestyle="none", linewidth=0.2, marker=r"$▦$", markersize=20, color=t.get_colors())
                for t in self.ax.collections[:] if type(t) is LineCollection
            ]
            labels = [t.get_label() for t in self.ax.collections[:]]
            line_legend_handles = [lines[0]]
            line_legend_labels = [labels[0]]
            self.handles += line_legend_handles
            self.labels += line_legend_labels

    def create_mapview(self):
        return flopy.plot.PlotMapView(model=self.model, layer=0)

    def plot_grid(self):
        self.mapview.plot_grid(label="Модельная сетка", colors='black', linewidth=0.3, facecolor="white", alpha=0.4)

    def get_heads(self):
        head_file = os.path.join(newpth, f"{self.model.name}.hds")
        head = flopy.utils.HeadFile(head_file)
        print(head.get_data(kstpkper=self.kper))
        return head.get_data(kstpkper=self.kper if self.kper else (0, 0))

    def get_npf(self):
        return self.model.npf.k.array[self.layer]

    def get_rch(self):
        return self.model.rch.recharge.array

    def plot_array(self):
        cmap = {"hk": "Spectral", "heads": "Blues", "rch": "BuPu"}
        data = None
        if self.arr_plot == "heads":
            data = self.get_heads()[self.layer]
        elif self.arr_plot == "hk":
            data = np.log10(self.get_npf())
        elif self.arr_plot == "rch":
            data = self.get_rch()
        if data is not None:
            pltview = self.mapview.plot_array(a=data, alpha=0.7, cmap=cmap.get(self.arr_plot, "Blues"))
            cbar = plt.colorbar(pltview, shrink=0.75, orientation="horizontal", pad=0.01)
            if self.arr_label:
                cbar.ax.set_xlabel(f'{self.arr_label}')
            else:
                cbar.ax.set_xlabel(f'Уровни {self.layer + 1} слоя, м')

    def plot_cnt(self, cnt):
        if self.base_layers.get("grid"):
            with rasterio.open(cnt.get("path"), crs=self.epsg) as src:
                grid = gpd.read_file(self.base_layers.get("grid"))
                grid.crs = self.epsg
                grid = grid.to_crs(src.crs)
                out_image, out_transform = mask(src, grid.geometry, crop=True)
                out_image = np.ma.masked_array(out_image, mask=(out_image == src.nodata))
                out_meta = src.meta.copy()
                print(out_image)
            show(out_image, ax=self.ax, transform=out_transform, contour=True, **cnt.get("options"))
            self.cnt_handles(cnt.get("options"), cnt.get("label"))
        else:
            print("Не указан грид файл в base_layers")

    def cnt_handles(self, cnt_options, label):
        marker_options = {"marker": "", "linewidth": 1, "linestyle": '-', 'color': cnt_options.get("colors")}
        self.handles += [Line2D([0], [0], **marker_options)]
        self.labels += [label]

    def get_res(self):
        wells_df = pd.read_csv(os.path.join(self.output_path, "vectors", "residuals.csv"))
        return gpd.GeoDataFrame(wells_df, geometry=gpd.points_from_xy(wells_df['x'], wells_df['y']))

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

    def plot_res(self):
        cmap_res = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
        wells_gdf = self.get_res()
        ax_res = wells_gdf.plot(ax=self.ax, column="residual", legend=True, cmap=cmap_res, scheme='quantiles', k=10,
                                edgecolors='black')
        self.res_legend(ax_res)

    def wrap_legend(self):
        self.labels = [self.labels[0]] + ['\n'.join(wrap(l, 30)) for l in self.labels[1:]]

    def create_legend(self):
        leg = self.ax.legend(handles=self.handles, labels=self.labels, loc='upper left', bbox_to_anchor=(1.04, 1),
                             frameon=False,
                             prop={'size': 13})

    def plot(self, annotations=None):
        if self.plot_grid_ins:
            self.plot_grid()
        self.base_handle_labels()
        if self.base_object:
            gdf = self.get_baselayer(self.base_object.get("path"))
            gdf.plot(ax=self.ax, **self.base_object.get("options"))
            self.other_handles(type=self.base_object.get("type_legend"),
                               label=self.base_object.get("label"),
                               marker_options=self.base_object.get("marker_options"))
        self.add_basemap()
        if self.arr_plot:
            self.plot_array()
        if self.contours:
            for cnt in self.contours:
                self.plot_cnt(cnt)
        if base_layers:
            for blay, blay_inf in base_layers.items():
                if blay != "grid":
                    gdf = self.get_baselayer(blay_inf.get("path"))
                    if blay == "riv":
                        self.plot_river(gdf)
                        self.river_handles(label=blay_inf.get("label"))
                    if blay == "drn":
                        self.plot_drn(gdf)
                        self.drn_handles()
                    else:
                        if blay_inf.get("options"):
                            gdfplot = gdf.plot(ax=self.ax, **blay_inf.get("options"))
                            if blay_inf.get("label"):
                                if blay_inf["options"].get("column"):
                                    pass
                                else:
                                    self.other_handles(type=blay_inf.get("type_legend"),
                                                       label=blay_inf.get("label"),
                                                       marker_options=blay_inf.get("marker_options"))
                        else:
                            gdf.plot(ax=self.ax)
        if self.residuals:
            self.plot_res()

        hk_pars = pst.parameter_data.loc[pst.parameter_data.pargp == f"pg4"]
        par_sum = sc.get_parameter_summary().sort_values("percent_reduction", ascending=False)
        hk_parsum = par_sum.loc[hk_pars.parnme]
        hk_parsum[['x', 'y']] = hk_pars.loc[hk_parsum.index.values, ['x', 'y']]
        gdf = gpd.GeoDataFrame(hk_parsum, geometry=gpd.points_from_xy(hk_parsum.x, hk_parsum.y))
        # gdf.plot(column="percent_reduction", legend=True, ax=npf.ax)
        ax_res = gdf.plot(ax=self.ax, column="percent_reduction", markersize=gdf.percent_reduction * 10, legend=True,
                          scheme='natural_breaks', k=7, edgecolors='black', )
        legend = ax_res.get_legend()
        for ea in legend.legend_handles:
            ea.set_markeredgecolor('black')
            ea.set_markeredgewidth(1)
        res_handles, res_labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
        self.ax.get_legend().remove()
        title_res = "Процент влияния КФ 3 слоя на прогноз"
        title_res_proxy = Line2D([], [], color="none", label=title_res)
        self.handles += [title_res_proxy] + res_handles
        self.labels += [title_res] + res_labels

        if annotations:
            for k, v in annotations.items():
                self.ax.annotate(**v.get("options"))
                self.handles += [v.get("handle")]
                self.labels += [v.get("label")]
        self.wrap_legend()
        self.create_legend()
        output_folder = os.path.join(self.output_path, "pics")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output = os.path.join(output_folder, f"{self.name_pic}.png")
        plt.savefig(output, dpi=200, bbox_inches='tight')