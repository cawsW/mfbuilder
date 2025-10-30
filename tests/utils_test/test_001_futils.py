import os
from osgeo import osr
from mdlbuilder.futils import rename_and_reproject, get_osm, get_srtm, get_stream_net

srs = osr.SpatialReference()
srs.ImportFromWkt(
   'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)
proj_string = srs.ExportToProj4()
# reprojected_file = rename_and_reproject("./test_001_data.json", proj_string)
# get_osm("../raspad/inputs/vector/border_ex_buf.shp", "../raspad/inputs/vector/rivers_osm.shp", proj_string, 'river')
# get_osm("../pest_copy/inputs/vector/border_ex_buf.shp", "../pest_copy/inputs/vector/quarry_osm.shp", proj_string, 'quarry')
# get_stream_net(os.path.join("..", "raspad", "inputs", "raster", "dem.tif"), 5957681.09,15571232.33, os.path.join("..", "raspad", "inputs", "river3_by_sheds.geojson"))
get_srtm("../../dto/danilovka/input/vector/border_buf_wgs.shp", "../../dto/danilovka/input/raster/dem.tif", proj_string)
