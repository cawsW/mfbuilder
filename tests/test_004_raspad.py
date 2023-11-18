import os
from osgeo import osr
from mdlbuilder.mdlbuilder import ModelBuilder
import pandas as pd
import numpy as np
import geopandas as gpd
from plpygis import Geometry
import flopy.export.utils as flp_exp

srs = osr.SpatialReference()
srs.ImportFromWkt(
    'PROJCS["Pulkovo_1942_GK_Zone_15",GEOGCS["GCS_Pulkovo_1942",DATUM["D_Pulkovo_1942",SPHEROID["Krasovsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",15500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",87.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
proj_string = srs.ExportToProj4()

config_model = {
    "base": {
        "name": "rsp",
        "workspace": "raspad",
        "exe": "../bin/mf6",
        "version": "mf6",
        "units": "DAYS",
        "steady": True,
    },
    "grid": {
        "type": "unstructured",
        "buffer": True,
        "boundary": os.path.join("inputs", "vector", "border_ex.shp"),
        "proj_string": proj_string,
        "cell_size": 150,
        "nlay": 3,
        "top": os.path.join("inputs", "raster", "dem.tif"),
        "botm": [os.path.join("inputs", "raster", "bot_1.tif"), {"thick": 100}, {"thick": 200}],
        "min_thickness": 4,
        "gridgen_exe": "../bin/gridgen",
        "refinement": {
            "polygon": [(os.path.join("inputs", "vector", "drain.shp"), 2)]
        }
    },
    "sources": {
        "riv":
            {0: [{"stage": "top", "cond": "cond", "depth": 3,
                  "geometry": os.path.join("inputs", "vector", "rivers.shp"), "bname": "zn", "layers": [1]}]},
    },
    "parameters": {
        "npf": {"k": [4, 0.5, 0.001], "k33": [1/2, 1/5, 1/10], "icelltype": [1, 0, 0]},
        "rch": 0.0003,
        "ic": {"head": 450},
    },
    "observations": {
        "wells": [{"data": os.path.join("inputs", "vector", "heads_insitu_upd.csv"), "layers": 2, "name": "name"}],
    }
}
a = ModelBuilder(config_model, external=True)
# flp_exp.model_export("data.shp", a.base.model)
a.run()
a.export()

obs_prp = pd.read_csv(os.path.join("raspad", "inputs", "vector", "heads_insitu_upd.csv"))
obs_prp["well"] = "H_" + obs_prp["name"].astype(str)
obs_prp["time"] = 1.0
table_obs = pd.pivot_table(obs_prp, values='head', index='time',
                       columns=['well'], aggfunc=np.min)
table_obs = table_obs.reset_index()
table_obs.to_csv(os.path.join("raspad", "rsp.obs.head.csv"), index=False)
#
# import numpy as np
# from shapely import Point
# from mdlbuilder.mdlutils import generate_points_within_polygon
# boundary = gpd.read_file(os.path.join("raspad", "inputs", "vector", "border_ex.shp"))
# polygon = boundary.geometry.values[0]
# points = pd.read_csv(os.path.join("raspad", "inputs", "vector", "heads_insitu_upd.csv"))
# existing_points = [(x, y) for x, y in zip(points.x, points.y)]
# new_points = np.array(generate_points_within_polygon(polygon, existing_points, grid_spacing=2500, min_distance=500))
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
# points_in_polygon.to_file(os.path.join("raspad", "inputs", "vector", 'pp.shp'))