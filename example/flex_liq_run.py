import numpy as np
import pandas as pd
import geopandas as gpd
from joblib import Parallel, delayed
import sys
from regional_mh_risk.simple_liquefaction import setup_cpt, assign_gwt, liqcalc_cpts, make_gdf2, run_lpi_sgs
from regional_mh_risk.simulations import make_part_gdf 
from regional_mh_risk.preprocess import setup_grid
print('Imported packages')
print('Imported functions')

inputdir = './ground_motions/USGS_CPT_data/'
cpts = setup_cpt(inputdir)

cpts = assign_gwt(cpts,'./gw_tifs/')

print('Set up cpts')

mags = np.genfromtxt('./M.csv',delimiter=',')
nsim = 2423 # or len(mags) or smaller number as needed
pgas = pd.read_csv('./pgas.csv',delimiter=',', index_col=0)

run_number = int(sys.argv[1])
minsim = run_number * 200
maxsim = 200 + run_number * 200
if run_number == 11:
   maxsim = nsim
print('min sim:',minsim)
print('max sim:',maxsim)

slr = [float(f"{float(sys.argv[2]):.2f}")]
SLR = [str(item) for item in slr]
Kh = 1.0
fun = np.loadtxt('./fun.csv', delimiter=',')
C_FC = np.loadtxt('./cfc.csv', delimiter=',')

lpis = {}
lpis = Parallel(n_jobs=-1,require='sharedmem',max_nbytes=None)(delayed(liqcalc_cpts)(sim = i, lpi = lpis, fun = fun, store = cpts,slr = slr, mags= mags, pgas = pgas, C_FC = C_FC, SLR = SLR,Kh = Kh) for i in range(minsim,maxsim))

print('Ran LPI')
dflpi = pd.DataFrame(lpis[0])

gdflpi = make_gdf2(dflpi,cpts)

points= setup_grid(utmX0=558500, utmY0=4172400, utmX1=570200, utmY1=4183800, width=100, geoplot='./inputs/alameda_city.geojson')

print('Created points')
points.drop(columns='index_right',inplace=True)

cpt_df = pd.DataFrame.from_dict(cpts, orient='index')
cpt_gdf = gpd.GeoDataFrame(cpt_df, geometry=gpd.points_from_xy(cpt_df.Lon, cpt_df.Lat),crs='EPSG:4326')

# separate artificial fill and dune sand
geologydir = '//home/groups/bakerjw/emongold/Liquefaction/'
shapefile_path = geologydir + 'deposits_shp/sfq2py.shp'
data = gpd.read_file(shapefile_path)
data.crs = 'EPSG:4326'
joined = gpd.sjoin(points,data,how="left",op="within")
points_af = joined[joined['PTYPE'] != 'Qds']
points_ds = joined[joined['PTYPE'] == 'Qds']
cpt_join = gpd.sjoin(cpt_gdf,data,how="left",op="within")
cpt_af = cpt_join[cpt_join['PTYPE'] != 'Qds']
cpt_ds = cpt_join[cpt_join['PTYPE'] == 'Qds']

gdflpi_af = make_part_gdf(dflpi,cpt_af)  # in simulations.py
gdflpi_ds = make_part_gdf(dflpi,cpt_ds)
for col in gdflpi_ds.columns:
    if col != 'geometry':
        if gdflpi_ds[col].sum() == 0:
            gdflpi_ds[col][0] = 1e-6
            gdflpi_ds[col][3] = 1e-6

points_ds = Parallel(n_jobs=-1,require='sharedmem',max_nbytes=None)(delayed(run_lpi_sgs)(sim = i, points = points_ds,gdflpi = gdflpi_ds) for i in range(minsim,maxsim))
points_af = Parallel(n_jobs=-1,require='sharedmem',max_nbytes=None)(delayed(run_lpi_sgs)(sim = i, points = points_af,gdflpi = gdflpi_af) for i in range(minsim,maxsim))
points = pd.concat([points_af[0],points_ds[0]])
# save points as a .csv
savename = str(maxsim) + 'points' + SLR[0]
points.to_csv(savename + '.csv', index=False)

print('all done')
