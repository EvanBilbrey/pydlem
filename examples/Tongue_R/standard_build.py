import sqlite3
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import xarray as xr

from prep.datafile import CreateInputFile
from prep.datafile import check_format
from prep.lakegeom import area_from_eac_curve
from prep.lakegeom import calc_lake_depth
from prep.lakegeom import calc_fetch_length
from prep.metdata import calculate_vpd

# add spatialite to path variable
spatialite_pth = 'C:/Users/CNB968/Spatialite'
os.environ['PATH'] = spatialite_pth + ';' + os.environ['PATH']
# connect to sqlite db and enable spatialite
conn = sqlite3.connect(r'C:\Users\CNB968\OneDrive - MT\Tongue River\RSM Data\Reservoir_Data\reservoirs_test.sqlite')
conn.enable_load_extension(True)
conn.load_extension("mod_spatialite")
# Query reservoir locations from polygon centroids
sql = "SELECT permanent_, AsBinary(Geometry) AS geom FROM reservoirs;"
resvs = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col='geom')
conn.close()
# get centroids & Location IDs
pnts = resvs['geom'].centroid
res_pnts = gpd.GeoDataFrame({
    'Location_ID': resvs['permanent_']},
    geometry=pnts)
coordinates = zip(pnts.x.to_list(), pnts.y.to_list())

T = CreateInputFile(coords=res_pnts, res_data=20.0)

sql = "SELECT permanent_, AsBinary(Centroid(Geometry)) AS geom FROM reservoirs;"
cur = conn.execute("SELECT permanent_, AsText(Centroid(Geometry)) AS coords FROM reservoirs WHERE permanent_ = ?;", ('71772797',))
conn.row_factory = sqlite3.Row
cur = conn.execute("SELECT permanent_, AsBinary(Centroid(Geometry)) AS geom FROM reservoirs;")
res = cur.fetchall()

gpd.from_postgis

conn.close()

## Load netcdf from file and create datafile
nc_path = Path(r'C:\Users\CNB968\OneDrive - MT\GitHub\pydlem\examples\Tongue_R\met_datafile_ex.nc')
mod_db_path = Path(r'C:\Users\CNB968\OneDrive - MT\Modeling\Riverware\Models\Tongue River\TRWMM.db')
conn = sqlite3.connect(mod_db_path)
sql = 'SELECT * FROM ReservoirData WHERE FID BETWEEN ? AND ?'
resdata = pd.read_sql(sql, conn, params=(56, 60), index_col='datetime', parse_dates=True)
eac_curves = pd.read_sql("SELECT * FROM ReservoirEACs WHERE FID BETWEEN ? AND ?", conn, params=(56, 60))
conn.close()
resdata = resdata.set_index('FID', append=True)
resids = resdata.index.unique(level='FID')
Alst = []
for i in resids:
    curve = eac_curves.loc[eac_curves['FID'] == i].iloc[:, 0:3]
    indat = resdata.xs(i, level='FID', drop_level=False)['Storage_acreft']
    lakeA = area_from_eac_curve(indat, curve)
    Alst.append(lakeA)
LA = pd.concat(Alst)

ex_dset = xr.open_dataset(nc_path)
sdset = ex_dset.sel(location=['71772129', '71759808', '71818121', '120031076', '71772901'])


DS = CreateInputFile(None, )

mod_db_path = Path(r'C:\Users\CNB968\OneDrive - MT\Modeling\Riverware\Models\Tongue River\TRWMM.db')
conn = sqlite3.connect(mod_db_path)
sql = 'SELECT * FROM ReservoirData WHERE FID IN (?, ?)'
#conn.row_factory = sqlite3.Row
df = pd.read_sql(sql, conn, params=(56, 57))
conn.close()

import pynhd as nhd
import pygridmet

from chmdata.thredds import GridMet

from prep.metdata import get_gridmet_at_points



t = nhd.WaterData('wbd06').byid('huc6', '100901')
tc = t.centroid
coords = (tc.geometry.x[0], tc.geometry.y[0])

test = get_gridmet_at_points(coords=[coords], loc_ids=['100901'], end='2024-05-01')

geometry = t.geometry[0]

var = ["pr", "tmmn"]
dates = ("2000-01-01", "2000-06-30")

daily = pygridmet.get_bygeom(geometry, dates, variables=var, snow=False)
daily

dates = ("2000-01-01", "2006-12-31")
data = pygridmet.get_bycoords(coords, dates, variables=var)

data = GridMet('pr', start='1979-01-01', end='2023-12-31', lat=coords[1], lon=coords[0])

####### Run Model Testing
nc_path = Path(r'C:\Users\CNB968\OneDrive - MT\GitHub\pydlem\examples\tr2100_test_datafile.nc')
ex_dset = xr.open_dataset(nc_path)

check_format(ex_dset)
