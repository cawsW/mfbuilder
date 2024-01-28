import os
import geopandas as gpd
from osgeo import osr
import pandas as pd
import numpy as np
from shapely.geometry import Point
from mdlbuilder.interpolation import lines_to_points, prepare_point_df, kriging_interpolate, create_raster, find_anomaly, create_variogram

srs = osr.SpatialReference()
srs.ImportFromWkt(
    'PROJCS["unknown",GEOGCS["GCS_unknown",DATUM["D_Unknown_based_on_Krassovsky_1942_ellipsoid",SPHEROID["Krassovsky_1942",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",2250000.0],PARAMETER["False_Northing",-5714743.504],PARAMETER["Central_Meridian",44.55],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
proj_string = srs.ExportToProj4()

path = "../dzerginsk/input/vector"
out_path = "../dzerginsk/input/raster"
source_path = "../novgorod/input/vector"

boundary = gpd.read_file(os.path.join(path, "envelope.shp"))
CRS = boundary.crs
bounds = boundary.bounds
grid_x = np.linspace(bounds.minx[0], bounds.maxx[0], num=500)
grid_y = np.linspace(bounds.miny[0], bounds.maxy[0], num=500)
aoiGeom = boundary.loc[0].geometry
# N-Q
nqbot_gdf = gpd.read_file(os.path.join(path, 'bot_q_line.shp'), crs=proj_string)
nqbot_abs_gdf = gpd.read_file(os.path.join(path, 'bot_q_point.shp'), crs=proj_string)

lines_df = lines_to_points(nqbot_gdf)
# nqbot_abs_gdf = nqbot_abs_gdf[nqbot_abs_gdf["VG"].str.contains("aQl-lV")]
nqbot_abs_gdf = nqbot_abs_gdf[nqbot_abs_gdf["VG"].str.contains("-aQ")]
points_df = prepare_point_df(nqbot_abs_gdf, "ABS_bot")
result = pd.concat([lines_df, points_df])
# result = points_df
result["elevation"] = result["elevation"].astype(float)
result = gpd.GeoDataFrame(
    result,
    geometry=[Point(xy) for xy in zip(result["X"], result["Y"])],
    crs=proj_string
)
points_in_polygons = gpd.sjoin(result, boundary, how="inner", op="within")[["X", "Y", "elevation"]]

print(points_in_polygons)
# _, normal_df = find_anomaly(result)
# points_in_polygons = points_in_polygons.sample(600)
var = create_variogram(points_in_polygons, "gaussian").parameters
pars = {'sill': var[1], 'range': var[0], 'nugget': var[2]}
elev, errors = kriging_interpolate(points_in_polygons, "power", grid_x, grid_y, pars)
create_raster(os.path.join(out_path, "bot_q.tif"), elev, grid_x, grid_y, CRS)
print("n-q")