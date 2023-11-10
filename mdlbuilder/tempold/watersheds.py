import os
import elevation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pysheds.grid import Grid
import geopandas as gpd
import fiona

def crs_wgs(gdf):
    if gdf.crs != 'EPSG:4326':
        return gdf.to_crs('EPSG:4326')
    return gdf

boundary = gpd.read_file('./novgorod/input/vectors/boundary.shp')
boundary_wgs = crs_wgs(boundary)
buffer = 0.01
bounds = boundary_wgs.bounds.minx[0] - buffer, boundary_wgs.bounds.miny[0] - buffer, boundary_wgs.bounds.maxx[0] + buffer, boundary_wgs.bounds.maxy[0] + buffer
dem_path = '/novgorod/output.tif'
output = os.getcwd() + dem_path
elevation.clip(bounds=bounds, output=output, product='SRTM3')

dst_crs = boundary.crs  # Define the target coordinate system (here using WGS84)

output_raster = "./novgorod/output_crs.tif"
with rasterio.open(output) as src:
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(output_raster, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

grid = Grid.from_raster(output)
dem = grid.read_raster(output)
# Condition DEM
# ----------------------
#Fill pits in DEM
pit_filled_dem = grid.fill_pits(dem)

#Fill depressions in DEM
flooded_dem = grid.fill_depressions(pit_filled_dem)

# Resolve flats in DEM
inflated_dem = grid.resolve_flats(flooded_dem)
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

# Compute flow directions
# -------------------------------------
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
acc = grid.accumulation(fdir, dirmap=dirmap)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
# Specify pour point
# x, y = 88.16527972, 53.76126754

# x, y = 88.2529007, 53.8896653
x, y = 44.332940, 56.104766
# x, y = 88.0885877, 53.7556022
print(np.amax(acc))
# Snap pour point to high accumulation cell
x_snap, y_snap = grid.snap_to_mask(acc > 10000, (x, y))
# grid.viewfinder = fdir.viewfinder

# Find the row and column index corresponding to the pour point
# col, row = grid.nearest_cell(x, y)

# Delineate the catchment
# catch = grid.catchment(x=col, y=row, fdir=fdir, dirmap=dirmap, xytype='index')

# Delineate the catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,
                       xytype='coordinate')

# Crop and plot the catchment
# ---------------------------
# Clip the bounding box to the catchment
grid.clip_to(catch)
clipped_catch = grid.view(catch)


# Plot the catchment
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)
plt.show()
# Calling grid.polygonize without arguments will default to the catchment mask
shapes = grid.polygonize()


schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

with fiona.open('./novgorod/input/catchment.shp', 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1
