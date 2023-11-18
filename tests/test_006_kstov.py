import os
import re
from osgeo import osr
import flopy.export.utils as flp_exp
from mdlbuilder.mdlbuilder import ModelBuilder

srs = osr.SpatialReference()
srs.ImportFromWkt(
    'PROJCS["unknown",GEOGCS["GCS_unknown",DATUM["D_Unknown_based_on_Krassovsky_1942_ellipsoid",SPHEROID["Krassovsky_1942",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",2250000.0],PARAMETER["False_Northing",-5714743.504],PARAMETER["Central_Meridian",44.55],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
proj_string = srs.ExportToProj4()

config_model = {
    "base": {
        "name": "kstov",
        "workspace": "pest_copy",
        "exe": "../bin/mf6",
        "version": "mf6",
        "steady": True,
    },
    "grid": {
        "type": "structured",
        "buffer": False,
        "boundary": os.path.join("input", "vector", "boundary.shp"),
        "cell_size": 200,
        "nlay": 1,
        "top": os.path.join("input", "raster", "dem_new.tif"),
        "botm": [os.path.join("input", "raster", "bot_q.tif")],
        "proj_string": proj_string,
        "min_thickness": 7
    },
    # "sources": {
    #     "riv": {0: [
    #         {"stage": os.path.join("input", "raster", "river_stages_kstov.tif"), "cond": 30, "depth": 1.5,
    #          "geometry": os.path.join("input", "vector", "small_rivers.shp"), "bname": "zn_", "layers": [1]},
    #     ]},
    #     "drn": {0: [
    #         {"geometry": "all", "head": "top", "cond": 10000, "layers": [1]},
    #     ]},
    #     "chd": {0: [
    #         {"head": os.path.join("input", "raster", "river_stages_kstov.tif"), "layers": [1],
    #          "geometry": os.path.join("input", "vector", "zones_river.shp")},
    #     ]},
    #     "ghb": {
    #         0: [{"head": "head", "cond": 10, "geometry": os.path.join("input", "vector", "ghb.shp"), "layers": [1]}]
    #     },
    # },
    # "parameters": {
    #     "npf": {"k": [4], "icelltype": [1]},
    #     "rch": 0.00025,
    # },
    "observations": {
        "wells": [{"data": os.path.join("input", "vector", "well_regime_new.csv"), "layers": 1, "name": "name"}],
    }
}
a = ModelBuilder(config_model, external=True, editing=True)
a.run()
# flp_exp.model_export("data.shp", a.base.model)
a.export()
# import pandas as pd
# obs_prp = pd.read_csv(os.path.join("kstov", "input", "vector", "well_regime_new.csv"))
# obs_prp["well"] = "H_" + obs_prp["name"].astype(str)
# obs_prp["time"] = 1.0
# obs_res = pd.read_csv(os.path.join("kstov", "output", "residuals.csv"))
# obs_prp = obs_prp[obs_prp["well"].isin(obs_res["name"])]
# table_obs = pd.pivot_table(obs_prp, values='head', index='time',
#                        columns=['well'])
# table_obs = table_obs.sort_index(axis='columns', level='well', key=lambda col: col.map(lambda x: int(re.split('(\d+)',x)[-2])))
# table_obs = table_obs.reset_index()

# table_obs.to_csv(os.path.join("kstov", "kstov.obs.head.csv"), index=False)

# import numpy as np
# from shapely import Point
# from mdlbuilder.mdlutils import generate_points_within_polygon
# import geopandas as gpd
# from plpygis import Geometry
#
# boundary = gpd.read_file(os.path.join("kstov", "input", "vector", "boundary.shp"))
# polygon = boundary.geometry.values[0]
# points = pd.read_csv(os.path.join("kstov", "input", "vector", "well_regime_new.csv"))
# existing_points = [(x, y) for x, y in zip(points.x, points.y)]
# new_points = np.array(generate_points_within_polygon(polygon, existing_points, grid_spacing=5000, min_distance=1000))
# new_points = [Point(x, y) for x, y in zip(new_points[:, 0], new_points[:, 1])]
# points_gdf = gpd.GeoDataFrame(new_points, columns=['geometry'])
#
# points_in_polygon = points_gdf[points_gdf.geometry.apply(lambda point: boundary.geometry.contains(point).any())]
#
# points_in_polygon.crs = boundary.crs
# points_in_polygon["name"] = points_in_polygon.reset_index().index + 1
# points_in_polygon["name"] = 'pp_' + points_in_polygon["name"].astype(str)
# for index, pp in points_in_polygon.iterrows():
#     point = [Geometry(points_in_polygon.geometry.to_wkb()[index]).geojson["coordinates"]]
# points_in_polygon.to_file(os.path.join("kstov", "input", "vector", 'pp.shp'))