import json
import os.path

import elevation
import geopandas as gpd
import osmnx as ox
import numpy as np
from pysheds.grid import Grid
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def rename_and_reproject(configs, crs):
    """
    Rename and reproject a spatial data file.

    Parameters:
    new_epsg (int): The EPSG code of the new projection.

    """
    f = open(configs)
    data = json.load(f)
    for inputd, outputd in data.items():
        gdf = gpd.read_file(inputd)
        gdf = gdf.to_crs(crs=crs)

        path = os.path.dirname(outputd)
        pth_check = os.path.exists(path)
        if not pth_check:
            os.makedirs(path)

        gdf.to_file(outputd, encoding="utf-8")
    

def get_osm(extent, output, crs, tag="river"):
    """
    Extract river data from OSM based on the extent of a given shapefile.

    Parameters:
    extent (str): Path to the input extent.
    output (str): Path to save the extracted river data.
    """

    gdf = gpd.read_file(extent, crs=crs)
    gdf = gdf.to_crs(crs="EPSG: 4326")

    bbox = gdf.total_bounds
    north, south, east, west = bbox[3], bbox[1], bbox[2], bbox[0]
    if tag == "river":
        tags = {'waterway': 'river', 'natural': "water"}
    elif tag == "stream":
        tags = {'waterway': 'stream'}
    elif tag == "quarry":
        tags = {'landuse': 'quarry'}

    gdf_rivers = ox.geometries_from_bbox(north, south, east, west, tags)
    gdf_rivers = gdf_rivers.to_crs(crs)
    path, ext = os.path.splitext(output)
    if tag == "river":
        poly = gdf_rivers[gdf_rivers["geometry"].geom_type.isin(["Polygon"])][["name", "geometry"]]
        poly.to_file(f"{path}_poly{ext}", index=False, encode="utf-8")
    elif tag == "quarry":
        poly = gdf_rivers[gdf_rivers["geometry"].geom_type.isin(["Polygon"])][["geometry"]]
        poly.to_file(f"{path}_poly{ext}", index=False, encode="utf-8")
    elif tag in ["river", "stream"]:
        lines = gdf_rivers[gdf_rivers["geometry"].geom_type.isin(["LineString"])][["geometry"]]
        print(lines)
        lines.to_file(f"{path}_line{ext}", index=False, encode="utf-8")


def get_srtm(extent, output, crs):
    boundary = gpd.read_file(extent, crs=crs)
    boundary_wgs = boundary.to_crs("EPSG:4326")
    bounds = boundary_wgs.total_bounds
    basename = os.path.join(os.getcwd(), output)
    elevation.clip(bounds=bounds, output=basename, product='SRTM1')

    dst_crs = boundary.crs  # Define the target coordinate system (here using WGS84)

    with rasterio.open(output) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def get_stream_net(dem, x, y, output):
    grid = Grid.from_raster(dem)
    dem = grid.read_raster(dem)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    conditioned_dem = grid.resolve_flats(flooded_dem)
    flowdir = grid.flowdir(conditioned_dem)
    acc = grid.accumulation(flowdir)
    x_snap, y_snap = grid.snap_to_mask(acc > 5000, (y, x))
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=flowdir, xytype='coordinate')
    grid.clip_to(catch)
    streams = grid.extract_river_network(flowdir, acc > 500, dirmap=dirmap)
    def saveDict(dic, file):
        f = open(file, 'w')
        f.write(str(dic))
        f.close()

    # save geojson as separate file
    saveDict(streams, output)
