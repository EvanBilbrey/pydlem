from chmdata.thredds import GridMet
from chmdata.thredds import BBox
from datetime import datetime, timedelta
import py3dep
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from shapely.geometry import Point

from prep.utils import get_gridmet_cells

from config import GRIDMET_PARAMS

# TODO - Default date did go to previous day...seems inconsistent with THREDDS,
#   sometimes it creates a mismatch in the dates and variable series, NEED TO SEE
#   if this is fixable bug in chmdata.thredds or if the end date needs an exception
#   in this script.
DEFAULT_DATES = ('1979-01-01', (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d"))


def get_gridmet_at_points(coords,
                          loc_ids,
                          start = DEFAULT_DATES[0],
                          end = DEFAULT_DATES[1],
                          crs = 4326) -> xr.Dataset:
    """
    Function takes a list of GridMET data parameters, start date, end date, and list of point coordinates
    and returns a discrete station formatted xarray dataset.
    :param params: the GridMET variables to download
    :param start: str "%Y-%m-%d" - Starting date of data extraction period
    :param end: str "%Y-%m-%d" = Ending date of data extraction period
    :param coords: list[tuple] - list of lat, lon coordinates where data will be extracted
    :return: an xarray dataset for discrete points (stations)
    """

    if isinstance(coords, list):
        coords = coords
    else:
        coords = [coords]

    if isinstance(loc_ids, list):
        location_ids = loc_ids
    else:
        location_ids = [loc_ids]

    loc_lat = []
    loc_lon = []
    loc_elev = py3dep.elevation_bycoords(coords, crs=crs)

    if isinstance(loc_elev, list):
        loc_elev = loc_elev
    else:
        loc_elev = [loc_elev]

    pnts = [Point(list(p)) for p in coords]
    pnt_gdf = gpd.GeoDataFrame({
        'pnt_id': location_ids,
        'geometry': pnts
    }, crs=crs)

    print("Retrieving GridMET cells...")
    gmt_cells = get_gridmet_cells(pnt_gdf)
    idx = [gmt_cells['pnt_id'].to_list().index(i) for i in location_ids]
    cellids = [gmt_cells['cell_id'].to_list()[i] for i in idx]
    unq_cells = np.unique(np.array(cellids))
    print("{0} unique GridMET cells found for {0} input points.".format(len(unq_cells), len(location_ids)))

    gmt_cntrs = gmt_cells.centroid

    pr = []
    srad = []
    th = []
    tmmn = []
    tmmx = []
    vs = []
    vpd = []

    cdsets = {}
    print("Fetching GridMET data for unique cells...")
    for cell in tqdm(unq_cells, desc='Cells'):
        clon = gmt_cntrs.iloc[cellids.index(cell)].x
        clat = gmt_cntrs.iloc[cellids.index(cell)].y
        datasets = []
        for p in GRIDMET_PARAMS:
            s = start
            e = end
            ds = GridMet(p, start=s, end=e, lat=clat, lon=clon).get_point_timeseries()
            datasets.append(ds)
        cdsets[cell] = datasets

    for i in range(len(coords)):
        c = coords[i]
        gmtcell_id = cellids[i]
        lon, lat = c
        loc_lat.append(lat)
        loc_lon.append(lon)

        dset = cdsets[gmtcell_id]

        pr.append(dset[GRIDMET_PARAMS.index('pr')])
        srad.append(dset[GRIDMET_PARAMS.index('srad')])
        th.append(dset[GRIDMET_PARAMS.index('th')])
        tmmn.append(dset[GRIDMET_PARAMS.index('tmmn')])
        tmmx.append(dset[GRIDMET_PARAMS.index('tmmx')])
        vs.append(dset[GRIDMET_PARAMS.index('vs')])
        vpd.append(dset[GRIDMET_PARAMS.index('vpd')])

    xds = xr.Dataset(
        {
            "precip": (['time', 'location'], pd.concat(pr, axis=1), {'standard_name': 'Precipitation',
                                                                     'units': 'mm'}),
            "min_temp": (['time', 'location'], pd.concat(tmmn, axis=1), {'standard_name': 'Minimum Temperature',
                                                                     'units': 'Kelvin'}),
            "max_temp": (['time', 'location'], pd.concat(tmmx, axis=1), {'standard_name': 'Maximum Temperature',
                                                                     'units': 'Kelvin'}),
            "srad": (['time', 'location'], pd.concat(srad, axis=1), {'standard_name': 'Downward Surface Shortwave Radiation',
                                                                     'units': 'W/m^2'}),
            "wind_dir": (['time', 'location'], pd.concat(th, axis=1), {'standard_name': 'Wind Direction',
                                                                     'units': 'Degrees Clockwise from N'}),
            "wind_vel": (['time', 'location'], pd.concat(vs, axis=1), {'standard_name': 'Wind Speed',
                                                                     'units': 'm/s'}),
            "vpd": (['time', 'location'], pd.concat(vpd, axis=1), {'standard_name': 'Vapor Pressure Deficit',
                                                                   'units': 'kPa'})
        },
        coords={
            "lat": (['location'], loc_lat, {'standard_name': 'latitude',
                                            'long_name': 'location_latitude',
                                            'units': 'degrees',
                                            'crs': '4326'}),
            "lon": (['location'], loc_lon, {'standard_name': 'longitude',
                                            'long_name': 'location_longitude',
                                            'units': 'degrees',
                                            'crs': '4326'}),
            "elev": (['location'], loc_elev, {'standard_name': 'elevation',
                                            'long_name': 'location_elevation',
                                            'units': 'meters'}),
            "location": (['location'], location_ids, {'long_name': 'location_identifier',
                                            'cf_role': 'timeseries_id'}),
            "time": pr[0].index
        },
        attrs={
            "featureType": 'timeSeries',
        }
    )

    return xds


def calculate_vpd(Tmin,
                  Tmax,
                  RHmin = None,
                  RHmax = None,
                  RHmean = None,
                  Tdew = None,
                  method='Tmin'):

    valid_methods = ['RHminmax', 'Tmin', 'Tdew', 'RHmean']

    if method not in valid_methods:
        raise ValueError("Method is invalid. Valid entries:{0}".format(valid_methods))

    etmin = 0.6108 * np.exp((17.27 * Tmin) / (Tmin + 237.3))
    etmax = 0.6108 * np.exp((17.27 * Tmax) / (Tmax + 237.3))
    es = (etmax + etmin) / 2.0

    if method == 'RHminmax':
        if (RHmin is None) or (RHmax is None):
            raise ValueError("Missing required humidity data inputs...")

        ea = (etmin * (RHmax / 100) + etmax * (RHmin / 100)) / 2.0

    if method == 'Tdew':
        if Tdew is None:
            raise ValueError("Missing Dew Point Temp data...")

        ea = 0.6108 * np.exp((17.27 * Tdew) / (Tdew + 237.3))

    if method == 'Tmin':
        ea = etmin

    if method == 'RHmean':
        if RHmean is None:
            raise ValueError("Missing required humidity data inputs...")

        ea = es * (RHmean / 100)

    vpd = es - ea

    return vpd

#def append_fetch_width_variable(met_dataset, wind_dir_name, )
