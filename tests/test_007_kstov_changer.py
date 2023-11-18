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
        "workspace": "kstov",
        "exe": "../bin/mf6",
        "version": "mf6",
        "steady": False,
        "perioddata": [(1.0, 1, 1.0), (60, 6, 1.0)],
    },
    "grid": {
        "type": "structured",
        "buffer": True,
        "boundary": os.path.join("input", "vector", "boundary.shp"),
        "cell_size": 200,
        "nlay": 1,
        "top": os.path.join("input", "raster", "dem.tif"),
        "botm": [os.path.join("input", "raster", "bot_q.tif")],
        "proj_string": proj_string,
        "min_thickness": 7
    },
    "parameters": {
        "sto": {"sy": 0.3, "ss": 0.0003, "iconvert": 1, "steady_state": [0], "transient": [1]}
    },
    "sources": {
        "chd": {0: [
            {"head": os.path.join("input", "raster", "river_stages_kstov_pred.tif"), "layers": [1],
             "geometry": os.path.join("input", "vector", "zones_flood_chd.shp")},
        ]},
    },
    "observations": {
        "wells": [{"data": os.path.join("input", "vector", "well_regime_new.csv"), "layers": 1, "name": "name"}],
    }
}
a = ModelBuilder(config_model, external=True, editing=True)
a.run()

# flp_exp.model_export("data.shp", a.base.model)
a.export()