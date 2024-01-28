import os
from osgeo import osr
from mdlbuilder.futils import rename_and_reproject, get_osm, get_srtm, get_stream_net

srs = osr.SpatialReference()
srs.ImportFromWkt(
   'PROJCS["Pulkovo_1942_GK_Zone_15",GEOGCS["GCS_Pulkovo_1942",DATUM["D_Pulkovo_1942",SPHEROID["Krasovsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",15500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",87.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'
)
proj_string = srs.ExportToProj4()
# reprojected_file = rename_and_reproject("./test_001_data.json", proj_string)
# get_osm("../raspad/inputs/vector/border_ex_buf.shp", "../raspad/inputs/vector/rivers_osm.shp", proj_string, 'river')
# get_osm("../pest_copy/inputs/vector/border_ex_buf.shp", "../pest_copy/inputs/vector/quarry_osm.shp", proj_string, 'quarry')
get_stream_net(os.path.join("..", "raspad", "inputs", "raster", "dem.tif"), 5957681.09,15571232.33, os.path.join("..", "raspad", "inputs", "river3_by_sheds.geojson"))
# get_srtm("../novgorod/input/vector/envelope.shp", "../novgorod/input/raster/dem.tif", proj_string)
