import os
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
import geopandas as gpd
import numpy as np
from osgeo import osr
import flopy
from flopy.utils import Raster
import stat

srs = osr.SpatialReference()
srs.ImportFromWkt(
    'PROJCS["unknown",GEOGCS["GCS_unknown",DATUM["D_Unknown_based_on_Krassovsky_1942_ellipsoid",SPHEROID["Krassovsky_1942",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",2250000.0],PARAMETER["False_Northing",-5714743.504],PARAMETER["Central_Meridian",44.55],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
proj_string = srs.ExportToProj4()

exe = os.path.join("../bin", "mf6")
st = os.stat(exe)
os.chmod(exe, st.st_mode | stat.S_IEXEC)
ws = "dzerginsk"
model_name = "dzerginsk"
sim = flopy.mf6.MFSimulation.load(sim_ws=ws, exe_name=exe)
gwf = sim.get_model(model_name)
modelgrid = gwf.modelgrid

raster_ws = os.path.join(ws,"input", "raster")
dem_name = "relief_NN_DZ_itog"

rio = Raster.load(os.path.join(raster_ws, dem_name))
dem_data = rio.resample_to_grid(
    modelgrid, band=rio.bands[0], method="nearest"
)

headfile = f"{model_name}.hds"
fname = os.path.join(ws, headfile)
hds = flopy.utils.HeadFile(fname)
head = hds.get_data(kstpkper=(5, 1))
head_change = dem_data - head

boundary = gpd.read_file('pest_copy/input/vector/boundary2.shp', crs=proj_string)
CRS = boundary.crs
bounds = boundary.bounds
head_change_rast = np.where(head_change <= 0.1, 0.1, head_change)
head_change_rast = np.where((head_change_rast > 0.1) & (head_change_rast<=0.7), 0.7, head_change_rast)
head_change_rast = np.where((head_change_rast > 0.7) & (head_change_rast<=1.2), 1.2, head_change_rast)
head_change_rast = np.where((head_change_rast > 1.2) & (head_change_rast<=2), 2, head_change_rast)
head_change_rast = np.where((head_change_rast > 2) & (head_change_rast<=3.2), 3.2, head_change_rast)
head_change_rast = np.where(head_change_rast > 3.2, 4, head_change_rast)
print(np.unique(head_change_rast))
range_x = bounds.maxx[0] - bounds.minx[0]
range_y = bounds.maxy[0] - bounds.miny[0]

# Calculate the number of cells for a grid cell size of 50m
num_cells_x = int(range_x / 50) 
num_cells_y = int(range_y / 50)

# cell2d = gwf.disv.cell2d.get_data()
# centroids_x = [cell[1] for cell in cell2d]
# centroids_y = [cell[2] for cell in cell2d]
cell2d = modelgrid.xyzcellcenters
centroids_x = cell2d[0].flatten()
centroids_y = cell2d[1].flatten()
# Define grid_x and grid_y
gridx, gridy = np.meshgrid(np.linspace(bounds.minx[0], bounds.maxx[0], num=num_cells_x),
                           np.linspace(bounds.miny[0], bounds.maxy[0], num=num_cells_y))
transform = from_origin(gridx.min(), gridy.max(), 50, 50)
# Original coordinates
original_coords = np.array([centroids_x, centroids_y]).T  # x_coordinates and y_coordinates should be 1D arrays containing the x and y coordinates of your original data points respectively.

# Interpolate data
interpolated_data = griddata(original_coords, head_change_rast.flatten(), (gridx, gridy), method='linear')  # 'nearest', 'linear', or 'cubic' interpolation methods can be used depending on the data.

# Now use interpolated_data with rasterio
new_dataset = rasterio.open(
    "flood_zones.tif",
    'w',
    driver='GTiff',
    height=interpolated_data.shape[0],
    width=interpolated_data.shape[1],
    count=1,
    dtype=str(interpolated_data.dtype),
    nodata=np.nan,
    crs=CRS,
    transform=transform,
)
new_dataset.write(interpolated_data[::-1], 1)
new_dataset.close()