from shapely import Polygon, LineString
from mfbuilder.mdlbuilder import ModelBuilder

k3 = []
mult = 0
for i in range(18):
    if i % 2 == 0:
        mult += 1
    k3.append(4e-5 * mult)

config_model = {
    "base": {
        "name": "hill",
        "workspace": "hill_ex1",
        "exe": "../bin/mf6",
        "version": "mf6",
        "units": "SECONDS",
        "steady": True,

    },
    "grid": {
        "type": "structured",
        "boundary": Polygon([(0, 0), (0, 18000), (18000, 18000), (18000, 0), (0, 0)]),
        "cell_size": 10,
        "nlay": 3,
        "top": 100,
        "botm": [50., 40., -10.],
    },
    "sources": {
        "riv": {
            0: [{"stage": "top", "cond": 100, "depth": 10, "geometry": [LineString([(0, 0), (0, 18000)])]}]
        },
        "ghb": {
            0: [{"head": 350, "cond": 1e-7, "geometry": [LineString([(18000, 0), (18000, 18000)])], "layers": [1, 3]}]
        },
    },
    "parameters": {
        "npf": {"k": [3e-4, 1e-7, 4e-5], "icelltype": [1, 0, 0]},
        "rch": [([Polygon([(0, 0), (0, 18000), (9000, 18000), (9000, 0), (0, 0)])], 63.072 * 3.170979e-10),
                ([Polygon([(9000, 0), (18000, 0), (18000, 18000), (9000, 18000), (9000, 0)])], 31.536 * 3.170979e-10)],
        "ic": {"head": 200},
    }
}
a = ModelBuilder(config_model)
a.run()
a.export()
