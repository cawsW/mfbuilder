import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import os
from matplotlib import cm
from rasterio.plot import show
from adjustText import adjust_text

output_conf = [
    {
        'raster_ws': os.path.join("rasters", "dz", "rch_lay_0.tif"),
        'output_ws': os.path.join("outputs", "dz", "rch_lay_0.png"),
        'name': 'w, м/сут',
        'min_val': 0.00001,
        'max_val': 0.001,
        'step_sub': 0.0001,
        'step_sup': 0.0001,
        'cmap_l': "cubehelix"
    },
    {
        'raster_ws': os.path.join("rasters", "dz", "npf_lay_0.tif"),
        'output_ws': os.path.join("outputs", "dz", "npf_lay_0.png"),
        'name': 'logK',
        'min_val': 0.1,
        'max_val': 5,
        'step_sub': 0.5,
        'step_sup': 1,
        'cmap_l': "twilight",
        "log": True
    },
    {
        'raster_ws': os.path.join("rasters", "dz", "head_15_lay_0_1.tif.tif"),
        'output_ws': os.path.join("outputs", "dz", "head_15_lay_0_1.png"),
        'add_ws': os.path.join("rasters", "dz", "dem_test_con.tif"),
        'add_vectors': os.path.join("vectors", "well_regime.csv"),
        'boundary': os.path.join("vectors", "boundary2.shp"),
        'name': 'Hдин а.о., м',
        'min_val': 50,
        'max_val': 150,
        'step_sub': 5,
        'step_sup': 15,
        'cmap_l': "Blues"
    },
    {
        'raster_ws': os.path.join("rasters", "dz", "head_00_lay_0.tif"),
        'output_ws': os.path.join("outputs", "dz", "head_00_lay_0.png"),
        'name': 'Hст а.о., м',
        'min_val': 50,
        'max_val': 150,
        'step_sub': 5,
        'step_sup': 5,
        'cmap_l': "Blues"
    }
]


def create_pic(raster_ws, output_ws, name, min_val, max_val, step_sub, step_sup, cmap_l, boundary=None, add_vectors=None, add_ws=None, log=False):
    src = rasterio.open(raster_ws, "r+")

    elevation_data = src.read()
    if log:
        elevation_data = np.where(elevation_data == -999, np.nan, np.log(elevation_data))
    else:
        elevation_data = np.where(elevation_data == -999, np.nan, elevation_data)

    levels = np.arange(min_val, max_val, step_sub)
    fig, ax = plt.subplots(1, figsize=(20, 20))
    cmap = cm.get_cmap(cmap_l, lut=20)

    map1 = show(elevation_data, transform=src.transform,ax=ax, cmap=cmap)
    show(elevation_data, transform=src.transform, contour=True, ax=ax, linewidths=0.7, colors='black',
                levels=levels,
                contour_label_kws=dict(inline=True, fontsize=10, levels=np.arange(min_val, max_val, step_sup)))
    if add_ws:
        src_dem = rasterio.open(add_ws, "r+")
        dem_data = src_dem.read()
        show(dem_data, transform=src_dem.transform, contour=True, ax=ax, linewidths=0.5, colors='#F4BF96',#'#F4BF96'
             levels=levels, alpha=0.7,
             contour_label_kws=dict(inline=True, fontsize=6, levels=np.arange(min_val, max_val, step_sup)))

    if add_vectors:
        vct = pd.read_csv(add_vectors)
        vct_g = gpd.GeoDataFrame(vct, geometry=gpd.points_from_xy(vct.x, vct.y))
        poly = gpd.read_file(boundary)
        points_within = gpd.sjoin(vct_g, poly, how='inner', predicate='within')
        points_within.plot(ax=ax, markersize=10, color='black')
        texts = []
        for i, x in points_within.iterrows():
            texts.append(ax.annotate(text=x['name_left'],
                                                  xy=(x.geometry.centroid.coords[0][0] + 100, x.geometry.centroid.coords[0][1] + 100,),
                                                  ha='left', va='top', fontsize=6))
        # points_within.apply(lambda x: ax.annotate(text=x['name_left'],
        #                                           xy=(x.geometry.centroid.coords[0][0] + 100, x.geometry.centroid.coords[0][1] + 100,),
        #                                           ha='left', va='top', fontsize=6), axis=1)
        adjust_text(texts)
    ax.ticklabel_format(useOffset=False, style='plain')
    im = map1.get_images()[0]
    colorbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, cmap=cmap)
    colorbar.set_label(name, rotation=90, labelpad=20)
    colorbar.ax.yaxis.set_label_position('left')

    fig.savefig(output_ws, bbox_inches='tight', pad_inches=0.1)


for output in output_conf:
    create_pic(**output)


