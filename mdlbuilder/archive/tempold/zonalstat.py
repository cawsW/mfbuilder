from osgeo import gdal
from osgeo import ogr
import os
import numpy as np
import csv


def boundingBoxToOffsets(bbox, geot):
    col1 = int((bbox[0] - geot[0]) / geot[1])
    col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
    row1 = int((bbox[3] - geot[3]) / geot[5])
    row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
    return [row1, row2, col1, col2]


def geotFromOffsets(row_offset, col_offset, geot):
    new_geot = [
        geot[0] + (col_offset * geot[1]),
        geot[1],
        0.0,
        geot[3] + (row_offset * geot[5]),
        0.0,
        geot[5]
    ]
    return new_geot


def setFeatureStats(fid, min, max, mean, median, sd, sum, count,
                    names=["min", "max", "mean", "median", "sd", "sum", "count", "id"]):
    featstats = {
        names[0]: min,
        names[1]: max,
        names[2]: mean,
        names[3]: median,
        names[4]: sd,
        names[5]: sum,
        names[6]: count,
        names[7]: fid,
    }
    return featstats



mem_driver = ogr.GetDriverByName("Memory")
mem_driver_gdal = gdal.GetDriverByName("MEM")
shp_name = "temp"

fn_raster = r"C:\Users\nurislamov\Desktop\Subsurface_fromModelBuilder\raster3395\Layer_1.bil.tif"
fn_zones = r"C:\Users\nurislamov\PycharmProjects\mf6moscow\Narvskaya\gridgen\gridlayer\layer_1.shp"

r_ds = gdal.Open(fn_raster)
p_ds = ogr.Open(fn_zones)

lyr = p_ds.GetLayer()
geot = r_ds.GetGeoTransform()
nodata = r_ds.GetRasterBand(1).GetNoDataValue()

zstats = []

p_feat = lyr.GetNextFeature()
niter = 0

while p_feat:
    if p_feat.GetGeometryRef() is not None:
        if os.path.exists(shp_name):
            mem_driver.DeleteDataSource(shp_name)
        tp_ds = mem_driver.CreateDataSource(shp_name)
        tp_lyr = tp_ds.CreateLayer('polygons', None, ogr.wkbPolygon)
        tp_lyr.CreateFeature(p_feat.Clone())
        offsets = boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(), \
                                       geot)
        new_geot = geotFromOffsets(offsets[0], offsets[2], geot)

        tr_ds = mem_driver_gdal.Create( \
            "", \
            offsets[3] - offsets[2], \
            offsets[1] - offsets[0], \
            1, \
            gdal.GDT_Byte)

        tr_ds.SetGeoTransform(new_geot)
        gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
        tr_array = tr_ds.ReadAsArray()

        r_array = r_ds.GetRasterBand(1).ReadAsArray( \
            offsets[2], \
            offsets[0], \
            offsets[3] - offsets[2], \
            offsets[1] - offsets[0])

        id = p_feat.GetFID()

        if r_array is not None:
            maskarray = np.ma.array( \
                r_array, \
                mask=np.logical_or(r_array == nodata, np.logical_not(tr_array)))

            if maskarray is not None:
                zstats.append(setFeatureStats( \
                    id, \
                    maskarray.min(), \
                    maskarray.max(), \
                    maskarray.mean(), \
                    np.ma.median(maskarray), \
                    maskarray.std(), \
                    maskarray.sum(), \
                    maskarray.count()))
            else:
                zstats.append(setFeatureStats( \
                    id, \
                    nodata, \
                    nodata, \
                    nodata, \
                    nodata, \
                    nodata, \
                    nodata, \
                    nodata))
        else:
            zstats.append(setFeatureStats( \
                id, \
                nodata, \
                nodata, \
                nodata, \
                nodata, \
                nodata, \
                nodata, \
                nodata))

        tp_ds = None
        tp_lyr = None
        tr_ds = None

        p_feat = lyr.GetNextFeature()

fn_csv = "zstats1.csv"
print(np.array([ i['min'] for i in zstats]))
col_names = zstats[0].keys()
with open(fn_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, col_names)
    writer.writeheader()
    writer.writerows(zstats)