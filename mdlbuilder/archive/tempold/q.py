import time, os
import psycopg2 as pg
from psycopg2.extensions import AsIs
from plpygis import Geometry
import numpy as np
import flopy.discretization as fgrid
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.interpolate import griddata
import skgstat as skg
import matplotlib.path as mplp

start_time = time.time()
path_save = 'rasters'
if not os.path.exists(path_save):
# Create a new directory because it does not exist
	os.makedirs(path_save)


def build_asc(point, method):
    data = []
    for elev, geom in point:
        g = Geometry(geom)
        data.append([g.x, g.y, float(elev)])
    data = np.array(data,dtype='float32')
    V = skg.Variogram(data[:, :2], data[:, 2], n_lags=20, model=method)
    v_param = V.parameters
    UK = OrdinaryKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model=method,
        nlags=20,
        coordinates_type='geographic',
        variogram_parameters=[v_param[1], v_param[0], v_param[2]]
    )
    z, ss = UK.execute("grid", gridx, gridy)
    z[~mask] = np.nan

    return z[::-1], ss[::-1]


def carb(condition):
    elevall = []
    errors = []
    for k, v in condition.items():
        print(k)
        if all(v.values()) and v['prev'] != 'change':
            cur.execute(query_thick, v)
            point = cur.fetchall()
            z, ss = build_asc(point, v['method'])
        else:
            if v['prev'] == 'change':
                cur.execute(query_change, v)
                point = cur.fetchall()
                z, ss = np.ones((gridy.shape[0], gridx.shape[0])) * np.float32(point[0][0]), np.ones(
                    (gridy.shape[0], gridx.shape[0]))
            else:
                cur.execute(query_bot, v)
                point = cur.fetchall()
                z, ss = build_asc(point, v['method'])
        elevall.append(np.array(z))
        errors.append(np.array(ss))
    return elevall, errors


# Connect to database
con = pg.connect(host='geoserver.mosecom.ru', port=5432, database='GEOL', user='NurislamovAI', password='556784')
cur = con.cursor()

# Create structure grid
cur.execute('select id_0,geom from model.\"ModelBord\"')
border = cur.fetchone()
g_area = Geometry(border[1])
xmin, ymin, xmax, ymax = g_area.bounds
delcell = 1000 / 111111
lx = xmax - xmin
ly = ymax - ymin
nrow = int(lx / delcell)
ncol = int(ly / delcell)
delc = delcell * np.ones(ncol, dtype=int)
delr = delcell * np.ones(nrow, dtype=int)
sgr = fgrid.StructuredGrid(
    delc, delr, top=None, botm=None, xoff=xmin, yoff=ymin, angrot=0, epsg=4326
)
gridx = np.array(sgr.xyzcellcenters[0][0],dtype='float32')
gridy = np.array(sgr.xyzcellcenters[1],dtype='float32')[:, 0][::-1]
X_rel, Y_rel = np.meshgrid(gridx, gridy)
mpath = mplp.Path(g_area.geojson['coordinates'][0][0][:-1])
points = np.array((X_rel.flatten(), Y_rel.flatten()),dtype='float32').T
mask = mpath.contains_points(points).reshape(X_rel.shape)


condition = {'bot_c1al-pr': {'prev': '4040', 'cur': '6100', 'next': '6200', 'method': 'spherical'},
             'bot_c2vr': {'prev': '4030', 'cur': '4040', 'next': '6100', 'method': 'spherical'},
             'bot_c2ks': {'prev': '4020', 'cur': '4030', 'next': '4040', 'method': 'spherical'},
             'bot_c2rst': {'prev': '4010', 'cur': '4020', 'next': '4030', 'method': 'spherical'},
             'bot_c2pd-mc': {'prev': None, 'cur': '4010', 'next': '4020', 'method': 'spherical'},
             'bot_c3kr': {'prev': '3040', 'cur': '4010', 'next': '4020', 'method': 'spherical'},
             'bot_c3ksm': {'prev': '3030', 'cur': '3040', 'next': '4010', 'method': 'spherical'},
             'bot_c3sc': {'prev': '3020', 'cur': '3030', 'next': '3040', 'method': 'spherical'},
             'bot_c3g-p1a': {'prev': '3010', 'cur': '3020', 'next': '3030', 'method': 'spherical'},
             'Layer_9': {'prev': 'change', 'cur': '3010', 'next': '3020', 'method': 'linear'},
             'bot_j2bt-k': {'prev': None, 'cur': AsIs('ANY(ARRAY[2040,2050])'), 'next': '3010', 'method': 'spherical'},
             'bot_j3v-k1al': {'prev': '2030', 'cur': AsIs('ANY(ARRAY[2040,2050])'), 'next': '3010',
                              'method': 'spherical'},
             'bot_k1al3': {'prev': '2020', 'cur': '2030', 'next': '2040', 'method': 'spherical'},
             'bot_k2': {'prev': '2010', 'cur': '2020', 'next': '2030', 'method': 'spherical'},
             'Layer_4': {'prev': 'change', 'cur': '2010', 'next': '2030', 'method': 'linear'},
             'bot_st-dns': {'prev': None, 'cur': AsIs('ANY(ARRAY[1510,1610])'), 'next': '2010', 'method': 'spherical'},
             'bot_dns': {'prev': None, 'cur': '1410', 'next': '1510', 'method': 'spherical'},
             'bot_dns-ms': {'prev': None, 'cur': '1310', 'next': '1410', 'method': 'spherical'},
             'bot_ms': {'prev': None, 'cur': '1230', 'next': '1310', 'method': 'spherical'},
             'bot_q': {'prev': None, 'cur': AsIs('ANY(ARRAY[1010,1020,1030,1040,1110,1210])'), 'next': '1230',
                       'method': 'spherical'},
             }

query_bot = """
select min(elev) as elev,geom from (select "Well_ID",lag(g."Hor_ID") OVER (PARTITION BY "Well_ID" ORDER BY "Bot_elev") AS prev_hor,
g."Hor_ID",
lead(g."Hor_ID") OVER (PARTITION BY "Well_ID" ORDER BY "Bot_elev") AS next_hor,
"Well_collar"-"Bot_elev" as elev,
(st_dump(w.geom)).geom as geom
 from "Wells_Geology" g left join "Wells" w using ("Well_ID") where "Well_collar" is not null) foo
 where "Hor_ID"=%(cur)s and next_hor is not null GROUP BY geom """

query_change = """
select max("Thickness") from
(select "Well_ID",
lag(g."Hor_ID") OVER (PARTITION BY "Well_ID" ORDER BY "Bot_elev") AS prev_hor,
g."Hor_ID",
lead(g."Hor_ID") OVER (PARTITION BY "Well_ID" ORDER BY "Bot_elev") AS next_hor,
Coalesce("Bot_elev" - lag("Bot_elev") OVER (PARTITION BY "Well_ID" ORDER BY "Well_ID", "Bot_elev"),
first_value("Bot_elev") OVER (PARTITION BY "Well_ID" ORDER BY "Well_ID", "Bot_elev")) AS "Thickness",
"Bot_elev",
(st_dump(w.geom)).geom as geom
from "Wells_Geology" g left join "Wells" w using ("Well_ID")) foo
where next_hor is not null and "Hor_ID"=%(cur)s
"""

query_thick = """select sum("Thickness"),geom from
(select "Well_ID",
lag(g."Hor_ID") OVER (PARTITION BY "Well_ID" ORDER BY "Bot_elev") AS prev_hor,
g."Hor_ID",
lead(g."Hor_ID") OVER (PARTITION BY "Well_ID" ORDER BY "Bot_elev") AS next_hor,
Coalesce("Bot_elev" - lag("Bot_elev") OVER (PARTITION BY "Well_ID" ORDER BY "Well_ID", "Bot_elev"),
first_value("Bot_elev") OVER (PARTITION BY "Well_ID" ORDER BY "Well_ID", "Bot_elev")) AS "Thickness",
"Bot_elev",
(st_dump(w.geom)).geom as geom
from "Wells_Geology" g left join "Wells" w using ("Well_ID")) foo
where prev_hor=%(prev)s and next_hor is not null and "Hor_ID"=%(cur)s GROUP BY geom"""

elevations, errors = carb(condition)
print('kriging is ok')
car_main_ind = 4
carb_main = elevations[4]
carb_elev_dpd = elevations[0:4]
carb_elev_upd = elevations[5:10]
carb_bot_dpd = np.array([np.subtract(carb_main, np.sum(elevations[i:car_main_ind], axis=0))
                         for i in range(0, car_main_ind)])
carb_bot_upd = np.array([(carb_main + np.sum(carb_elev_upd[:i], axis=0)) for i in
                         range(1, len(carb_elev_upd) + 1)])

mz_main = elevations[10]
mz_elev = elevations[11:15]
mz_bot = np.array([(mz_main + np.sum(mz_elev[:i], axis=0)) for i in range(1, len(mz_elev) + 1)])

q_bot = np.array([elev for elev in elevations[15:]])

print('form elevations')
query = """
select st_x(r.geom),st_y(r.geom),top from model.relief_points r inner join model."MapBord" m
on st_contains(m.geom,r.geom)
where top is not null;
"""

cur.execute(query)
point_rel = cur.fetchall()
data_rel = np.array(point_rel)
rel_z = griddata(data_rel[:, :2], data_rel[:, 2], (X_rel, Y_rel), method='linear')
print('relief is ok')
elev_postprocess = np.concatenate(
    (carb_bot_dpd, np.array([carb_main]), carb_bot_upd, np.array([mz_main]), mz_bot, q_bot, np.array([rel_z[::-1]])),
    axis=0)
# elev_postprocess = np.concatenate((carb_bot_dpd, np.array([carb_main]), carb_bot_upd), axis=0)
print(len(elev_postprocess))
#\\tech-geo\fgi\MODEL\bottom\errors
for i in range(len(elev_postprocess)):
    
    rasterCrs = CRS.from_epsg(4326)
	
    interpRaster = rasterio.open("./rasters/{0}.tif".format(str(list(condition)[i])),
                                 'w',
                                 height=carb_main.shape[0],
                                 width=carb_main.shape[1],
                                 count=1,
                                 dtype=carb_main.dtype,
                                 driver='GTiff',
                                 crs=rasterCrs,
                                 transform=from_origin(xmin, ymax, delcell, delcell), nodata=np.nan )
    z = np.minimum.reduce(elev_postprocess[i:])
    with open('./rasters/test.txt', 'w') as f:
    	f.write('Create a new text file!')
    interpRaster.write(z, 1)
    interpRaster.close()

    interperrors = rasterio.open("./rasters/errors_{0}.tif".format(str(list(condition)[i])),
                                 'w',
                                 height=carb_main.shape[0],
                                 width=carb_main.shape[1],
                                 count=1,
                                 dtype=carb_main.dtype,
                                 driver='GTiff',
                                 crs=rasterCrs,
                                 transform=from_origin(xmin, ymax, delcell, delcell))
    er = errors[i]
    interperrors.write(er, 1)
    interperrors.close()

print("--- %s seconds ---" % (time.time() - start_time))
