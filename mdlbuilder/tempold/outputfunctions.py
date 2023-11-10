import csv
import osgeo.ogr, osgeo.osr
from osgeo import ogr
from osgeo import gdal
import os
import numpy as np
import flopy.utils.binaryfile as bf
import pandas as pd
import flopy
import fiona
import re
from osgeo import osr
import rasterio
from osgeo.gdalconst import *
from rasterio.mask import mask


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def reproject(out_epsg, in_shp, out_shp, types):
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # get the input layer
    inDataSet = driver.Open(in_shp)
    inLayer = inDataSet.GetLayer()

    inSpatialRef = inLayer.GetSpatialRef()

    # loading projection
    sr = osr.SpatialReference(str(inSpatialRef))

    # detecting EPSG/SRID
    res = sr.AutoIdentifyEPSG()

    srid = sr.GetAuthorityCode(None)

    # input SpatialReference
    inSpatialRef.ImportFromEPSG(int(srid))

    # output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(out_epsg)

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # create the output layer

    if os.path.exists(out_shp):
        driver.DeleteDataSource(out_shp)

    outDataSet = driver.CreateDataSource(out_shp)
    outLayer = outDataSet.CreateLayer("Points_4326", geom_type=types)

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()

    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()

    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # destroy the features and get the next input feature
        outFeature.Destroy()
        inFeature.Destroy()
        inFeature = inLayer.GetNextFeature()

    # close the shapefiles
    inDataSet.Destroy()
    outDataSet.Destroy()
    inDataSet = None
    outDataSet = None


def createcsv(m, nlay, mfdir):
    npf = m.npf.k.array
    top=m.disv.top.array
    bot = m.disv.botm.array
    flow = m.wel.stress_period_data
    cell2d = m.disv.cell2d.array
    x = [i[1] for i in cell2d]
    y = [i[2] for i in cell2d]
    nodeflow = []
    flowrate = []
    layers = []
    h = bf.HeadFile(os.path.join(mfdir, 'dummy.hds'))
    heads = h.get_data()
    for i, j in flow.get_data().items():
        for k in j:
            nodeflow.append(k[0][1])
            flowrate.append(k[1])
            layers.append(k[0][0] + 1)
    rch = m.rcha.recharge.array

    dfflow = pd.DataFrame(list(zip(np.array(x)[nodeflow], np.array(y)[nodeflow], flowrate, layers)),
                          columns=['x', 'y', 'flow', 'layer'])
    dfflow.to_csv(os.path.join(mfdir, 'flow', 'flow.csv'), index=False)

    dfrch = pd.DataFrame(list(zip(x, y, rch[0][0])),
                         columns=['x', 'y', 'rch'])
    dfrch.to_csv(os.path.join(mfdir, 'rch', 'rch.csv'), index=False)

    for lay in range(nlay):
        df = pd.DataFrame(list(zip(x, y, heads[lay][0])), columns=['x', 'y', 'head'])
        df.to_csv(os.path.join(mfdir, 'head', 'Layer_' + str(lay + 1) + '_head.csv'), index=False)

    for lay in range(nlay):
        df = pd.DataFrame(list(zip(x, y, npf[lay])), columns=['x', 'y', 'k'])
        df.to_csv(os.path.join(mfdir, 'npf', 'Layer_' + str(lay + 1) + '_k.csv'), index=False)

    for lay in range(nlay+1):
        if lay==0:
            df = pd.DataFrame(list(zip(x, y, top)), columns=['x', 'y', 'z'])
            df.to_csv(os.path.join(mfdir, 'top', 'Layer_' + str(lay + 1) + '_top.csv'), index=False)
        else:
            df = pd.DataFrame(list(zip(x, y, bot[lay-1])), columns=['x', 'y', 'z'])
            df.to_csv(os.path.join(mfdir, 'top', 'Layer_' + str(lay+1) + '_top.csv'), index=False)


def csv_toshp(input_file, export_shp, EPSG_code, delimiter):
    spatialReference = osgeo.osr.SpatialReference()  # will create a spatial reference locally to tell the system what the reference will be
    spatialReference.ImportFromEPSG(int(EPSG_code))  # here we define this reference to be the EPSG code
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')  # will select the driver for our shp-file creation.
    shapeData = driver.CreateDataSource(export_shp)  # so there we will store our data
    layer = shapeData.CreateLayer('layer', spatialReference,
                                  osgeo.ogr.wkbPoint)  # this will create a corresponding layer for our data with given spatial information.
    layer_defn = layer.GetLayerDefn()  # gets parameters of the current shapefile
    index = 0

    with open(input_file, 'r') as csvfile:
        readerDict = csv.DictReader(csvfile, delimiter=delimiter)
        for field in readerDict.fieldnames:
            if field in ['flow', 'residual','zone']:
                new_field = ogr.FieldDefn(field, ogr.OFTReal)
                layer.CreateField(new_field)
            elif field in ['layer']:
                new_field = ogr.FieldDefn(field, ogr.OFTInteger)
                layer.CreateField(new_field)
            else:
                new_field = ogr.FieldDefn(field, ogr.OFTString)
                layer.CreateField(new_field)
        for row in readerDict:
            point = osgeo.ogr.Geometry(osgeo.ogr.wkbPoint)
            point.AddPoint(float(row['x']), float(row['y']))  # we do have LATs and LONs as Strings, so we convert them
            feature = osgeo.ogr.Feature(layer_defn)
            feature.SetGeometry(point)  # set the coordinates
            feature.SetFID(index)
            for field in readerDict.fieldnames:
                i = feature.GetFieldIndex(field)
                feature.SetField(i, row[field])
            layer.CreateFeature(feature)
            index += 1
    shapeData.Destroy()


def shp_totif(_out, _in, zfield):
    output = gdal.Grid(_out, _in, algorithm='linear:radius=0', zfield=zfield)
    # sr = osr.SpatialReference()
    # sr.ImportFromEPSG(3395)
    # # open your raster and set projection
    # ds = gdal.Open(_out, 1)  # the second argument (1) opens the raster in update mode
    # ds.SetProjection(sr.ExportToWkt())  # this method takes a string rather than an object
    # # save changes
    # del ds


def boundingBoxToOffsets(bbox, geot):
    col1 = int((bbox[0] - geot[0]) / geot[1])
    col2 = int((bbox[1] - geot[0]) / geot[1])
    row1 = int((bbox[3] - geot[3]) / geot[5])
    row2 = int((bbox[2] - geot[3]) / geot[5])
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
                    names=["мин.", "макс.", "сред.", "мед.", "стд.", "сумм.", "кол-во", "id"]):
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


def reprojectrast(input_raster, ouput_raster):
    gdal.Warp(ouput_raster, input_raster, dstSRS="EPSG:3395")


def masked(rast, zone):
    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"

    fn_raster = rast
    fn_zones = zone

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
    return zstats


def residual(nlay, modelWs, modelName, outputpath):
    obsOut = flopy.utils.observationfile.Mf6Obs(modelWs + '/' + modelName + '.obs.head.csv', isBinary=False)
    df = pd.DataFrame(zip(list(obsOut.obsnames), list(obsOut.data.tolist())[1:]), columns=['well', 'head'])
    obswell = []
    for lay in range(1, nlay + 1):
        with fiona.open(
                r"C:\Users\nurislamov\PycharmProjects\modelgms\shapes\wellsobs\layer_" + str(lay) + ".shp") as output:
            for feature in output:
                if feature['properties']['layer'] == str(lay) and feature['properties']['Watdepth'] != None:
                    if feature['properties']['Watdepth'] > 9 and feature['properties']['Watdepth'] < 1000:
                        if 'H_' + str(int(feature['properties']['Well_ID'])) in list(obsOut.obsnames):
                            obswell.append(
                                ['H_' + str(int(feature['properties']['Well_ID'])), feature['properties']['Watdepth'],
                                 feature['properties']['weight'], feature['properties']['layer'],
                                 feature['geometry']['coordinates'][0], feature['geometry']['coordinates'][1]])

    df2 = pd.DataFrame(obswell, columns=['well', 'head', 'weight', 'layer', 'x', 'y'])
    result = pd.merge(df, df2, how="left", on="well")
    result.head_y.fillna(result.head_x, inplace=True)
    result.weight.fillna(0.1, inplace=True)
    result[["x", "y"]] = result[["x", "y"]].apply(pd.to_numeric)
    result['residual'] = result.head_y - result.head_x
    result = result[result['x'].notna()]
    result.to_csv(os.path.join(outputpath, 'residual', 'residual.csv'), index=False)


def createstats(rstcat, zone, nmzone, dictlay):
    modbordtop = []
    for rast in sorted_alphanumeric(os.listdir(rstcat)):
        if rast.endswith('.tif') and 'Layer_0' not in rast:
            zstats = masked(os.path.join(rstcat, rast), zone)
            zstats[0]['Горизонт'] = dictlay[re.sub("^(\\D*\\d+).*", "\\1", rast)]
            zstats[0]['макс.' + nmzone] = zstats[0].pop('макс.')
            zstats[0]['мин.' + nmzone] = zstats[0].pop('мин.')
            zstats[0]['сред.' + nmzone] = zstats[0].pop('сред.')
            zstats[0]['сумм.' + nmzone] = zstats[0].pop('сумм.')
            modbordtop.append(zstats)
    dfmodbordtop = pd.DataFrame([item for sublist in modbordtop for item in sublist])
    return dfmodbordtop


def shpbylayer(dir, inshp, namepar, nlay):
    for lay in range(1, nlay + 1):
        with fiona.open(os.path.join(dir, inshp)) as input:
            meta = input.meta
            with fiona.open(
                    os.path.join(dir, "layer_" + str(lay) + "_" + str(namepar) + ".shp"),
                    'w',
                    **meta) as output:
                for feature in input:
                    if feature['properties']['layer'] == lay:
                        output.write(feature)


def cliprast(geomcrop, inraster, outraster, infolder, outfolder, outmain):
    # with fiona.open(geomcrop, "r") as shapefile:
    #     shapes = [feature["geometry"] for feature in shapefile]
    #
    # # read imagery file
    # with rasterio.open(os.path.join(outmain, infolder, inraster)) as src:
    #     out_image, out_transform = mask(src, shapes, crop=True, filled=True)
    #     out_meta = src.meta
    # # Save clipped imagery
    # out_meta.update({"driver": "GTiff",
    #                  "height": out_image.shape[1],
    #                  "width": out_image.shape[2],
    #                    })
    #
    # with rasterio.open(os.path.join(outmain, outfolder, outraster), "w", **out_meta) as dest:
    #     dest.write(out_image)
    gk500 = ogr.Open(
        geomcrop, 1)  # <====== 1 = update mode
    gk_lyr = gk500.GetLayer()

    for feature in gk_lyr:
        geom = feature.GetGeometryRef()
        if not geom.IsValid():
            feature.SetGeometry(geom.Buffer(0))  # <====== SetGeometry
            gk_lyr.SetFeature(feature)  # <====== SetFeature
            assert feature.GetGeometryRef().IsValid()  # Doesn't fail
            gk_lyr.ResetReading()
    gdal.Warp(os.path.join(outmain, outfolder, outraster), os.path.join(outmain, infolder, inraster),
              cutlineDSName=geomcrop)


def setnodata(fn):
    ds = gdal.Open(fn, 1)  # pass 1 to modify the raster
    n = ds.RasterCount  # get number of bands
    for i in range(1, n + 1):
        band = ds.GetRasterBand(i)
        arr = band.ReadAsArray()  # read band as numpy array
        arr = np.where(arr > 1e+10, -10000, arr)  # change 0 to -10000
        arr = np.where(arr == 0, -10000, arr)
        band.WriteArray(arr)  # write the new array
        band.SetNoDataValue(-10000)  # set the NoData value
        band.FlushCache()  # save changes
    del ds


def createcountor(rast, inter, dirout, outshp, nodata=-3.4028230607370965e+38):
    indataset1 = gdal.Open(rast, GA_ReadOnly)
    in1 = indataset1.GetRasterBand(1)
    dst_filename = outshp
    # Generate layer to save Contourlines in
    ogr_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(os.path.join(dirout, dst_filename))
    contour_shp = ogr_ds.CreateLayer('contour')

    field_defn = ogr.FieldDefn("ID", ogr.OFTInteger)
    contour_shp.CreateField(field_defn)
    field_defn = ogr.FieldDefn("elev", ogr.OFTReal)
    contour_shp.CreateField(field_defn)

    # Generate Contourlines
    gdal.ContourGenerate(in1, inter, 0, [], 1, nodata, contour_shp, 0, 1)
    ogr_ds = None
    del ogr_ds
