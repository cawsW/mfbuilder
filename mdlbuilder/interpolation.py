from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import IsolationForest
from skgstat import Variogram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin


def find_anomaly(df, contamiation=0.01):
    features = df[["X", "Y", "elevation"]]

    clf = IsolationForest(contamination=contamiation)  # contamination is the proportion of outliers in the data set

    clf.fit(features)

    # Predict the anomalies in the data
    pred = clf.predict(features)

    # The prediction returns '1' for a normal point and '-1' for an anomaly
    df['anomaly'] = pred
    anomalies = df[df['anomaly'] == -1]
    normal = df[df['anomaly'] != -1]
    return anomalies, normal


def plot_anomaly(nm_df, anomal_df, labels=False):
    fig, ax = plt.subplots()
    gen_df = pd.concat([nm_df, anomal_df])
    sc = ax.scatter(gen_df['X'], gen_df['Y'], c=gen_df['elevation'], cmap='viridis', label='Normal')
    ax.scatter(anomal_df['X'], anomal_df['Y'], c='red', label='Anomaly')

    if labels:
        for i in range(len(gen_df)):
            ax.annotate(gen_df['elevation'].iloc[i], (gen_df['X'].iloc[i], gen_df['Y'].iloc[i]))

    fig.colorbar(sc, ax=ax, label='Elevation')
    ax.legend()
    plt.show()


def create_variogram(df, model):
    points = df[['X', 'Y']].values.astype('float32')
    values = df['elevation'].values.astype('float32')
    variogram = Variogram(points, values, model=model)
    variogram.plot()
    plt.show()
    return variogram


def kriging_interpolate(df, model, gridx, gridy, par=None, nlags=25):
    print(par)
    OK = OrdinaryKriging(
        df['X'],
        df['Y'],
        df['elevation'],
        variogram_model=model,
        verbose=False,
        enable_plotting=True,
        # anisotropy_angle=90,
        pseudo_inv=True,
        nlags=nlags,
        # variogram_parameters=par if par else None
    )
    z, ss = OK.execute("grid", gridx, gridy)
    return z, ss


def create_raster(name, elevations, gridx, gridy, crs):
    transform = from_origin(gridx.min(), gridy.max(), np.diff(gridx)[0], np.diff(gridy)[0])
    new_dataset = rasterio.open(
        name,
        'w',
        driver='GTiff',
        height=elevations.shape[0],
        width=elevations.shape[1],
        count=1,
        dtype=str(elevations.dtype),
        crs=crs,
        transform=transform,
    )
    new_dataset.write(elevations[::-1], 1)
    new_dataset.close()


def lines_to_points(gdf):
    point_df = pd.DataFrame()
    gdf = gdf.explode()
    i = 0
    for index, row in gdf.iterrows():
        if row.geometry is not None:  # Check for false geometries
            try:
                # if row.geometry.disjoint(aoiGeom)==False:
                simp_row = row.geometry.simplify(300, preserve_topology=True)
                for coord in simp_row.coords:
                    # if Point(coord).within(aoiGeom): #Check if vertex is inside the polygon
                    point_df.loc[i, 'X'] = coord[0]
                    point_df.loc[i, 'Y'] = coord[1]
                    point_df.loc[i, 'elevation'] = row.Elevation
                    i += 1
            except ValueError:
                pass
    return point_df[["X", "Y", "elevation"]]


def prepare_point_df(gdf, elev_field):
    df = gdf.drop_duplicates(subset=["geometry"], keep='first')
    df["X"] = df.geometry.x
    df["Y"] = df.geometry.y
    df["elevation"] = df[elev_field]
    return df[["X", "Y", "elevation"]]