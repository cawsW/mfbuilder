import os
from osgeo import osr
from mdlbuilder.mdlbuilder import ModelBuilder

srs = osr.SpatialReference()
srs.ImportFromWkt(
    'PROJCS["unknown",GEOGCS["GCS_unknown",DATUM["D_Unknown_based_on_Krassovsky_1942_ellipsoid",SPHEROID["Krassovsky_1942",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",2250000.0],PARAMETER["False_Northing",-5714743.504],PARAMETER["Central_Meridian",44.55],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
proj_string = srs.ExportToProj4()

config_model = {
    "base": {
        "name": "nn",
        "workspace": "novgorod",
        "exe": "../bin/mf6",
        "version": "mf6",
        "steady": True,
    },
    "grid": {
        "type": "unstructured",
        "buffer": False,
        "gridgen_exe": "../bin/gridgen",
        "boundary": os.path.join("input", "vectors", "boundary.shp"),
        "cell_size": 1000,
        "nlay": 1,
        "top": os.path.join("input", "rasters", "Relief_new"),
        "botm": [os.path.join("input", "rasters", "bot.tif")],
        "proj_string": proj_string,
        "min_thickness": 7
        # "refinement": {
        #     "line": [(os.path.join("input", "vectors", "zones_small_river.shp"), 3)],
        #     "polygon": [(os.path.join("input", "vectors", "zones_river.shp"), 3),
        #                 (os.path.join("input", "vectors", "flood_zone.shp"), 2)],
        # }
    },
    "sources": {
        "riv": {0: [
            {"stage": "top", "cond": 100, "depth": 1.5,
             "geometry": os.path.join("input", "vectors", "zones_small_river.shp"), "bname": "zn_"},
            {"stage": "top", "cond": 1000, "depth": 5, "geometry": os.path.join("input", "vectors", "zones_river.shp"),
             "bname": "zn_"},
        ]},
        "drn": {0: [
            {"geometry": "all", "head": "top", "cond": 10000, "layers": [1]},
        ]},
        "wel": {0: [
            {"geometry": os.path.join("input", "vectors", "intakes.shp"), "q": "q_1", "layers": "lay"},
            {"geometry": os.path.join("input", "vectors", "test_data", "wells_test.shp"), "q": "q", "layers": "lay"},
        ]}
    },
    "parameters": {
        # "npf": {"k": [os.path.join("input", "rasters", "hk.tif")], "icelltype": [1]},
        # "rch": os.path.join("input", "rasters", "rch.tif"),
        "npf": {"k": [9], "icelltype": [1]},
        "rch": 0.0003,
    },
    "observations": {
        "wells": [{"data": os.path.join("input", "vectors", "well_regime_sort2.csv"), "layers": 1, "name": "id1"}],
    }
}
a = ModelBuilder(config_model)
a.run()
