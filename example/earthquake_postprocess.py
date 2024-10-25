# earthquake_postprocess.py to postprocess earthquake simulation results for Alameda case study
# by Emily Mongold, 2020-08-06

print('imports')
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import shapely.wkt
import os
from regional_mh_risk.simple_liquefaction import liq_frag_func
from regional_mh_risk.earthquake import eq_shaking_loss
from regional_mh_risk.postprocess import get_pM_pypsha
from NNR import NNR
import pickle

print('setup')
## savedir should contain the outputs of the liquefaction run
savedir = '/scratch/groups/bakerjw/emongold/all_alameda_geology/'
## outdir should contain the outputs of the ground shaking run
outdir = '/scratch/groups/bakerjw/emongold/new_gms/'

files = os.listdir(savedir)
bldgs = pd.read_csv('./bldgs_filtered_nsi.csv')

slrs = [0.0,0.25,0.5,0.75,1.0,1.25,1.5]
points = {}
for slr in slrs:
    csv_files = [file for file in files if file.endswith(str(slr) + '.csv')]
    for i in range(len(csv_files)):
        if i == 0:
            points[slr] = pd.read_csv(savedir + csv_files[i])
        else:
            file_path = os.path.join(savedir, csv_files[i])
            try:
                df = pd.read_csv(file_path).drop(columns=['utmX','utmY','lat','lon','Unnamed: 0','geometry'])
            except:
                df = pd.read_csv(file_path).drop(columns=['utmX','utmY','lat','lon','geometry'])
            points[slr] = pd.concat([points[slr], df], axis=1)
for slr in points.keys():
    points[slr]['geometry'] = points[slr]['geometry'].apply(lambda x: shapely.wkt.loads(x))
    points[slr].drop(columns=['index_right','utmX','utmY','geometry','id','AREA','PERIMETER','SFQ2_','SFQ2_ID','PTYPE','PTYPE2','LIQ','LIQ_SOURCE'],axis=1,inplace=True)

output = {}
sims = {}
for slr in points.keys():
    output[slr] = points[slr].drop(columns=['lat','lon'],axis=1)
    # change column names to integers of their name
    output[slr].columns = [int(i) for i in output[slr].columns]
    # sort output by columns 0-2423
    try: 
        output[slr] = output[slr].reindex(sorted(output[slr].columns), axis=1)
    except:
        # print duplicated columns
        print(slr)
        print(output[slr].columns[output[slr].columns.duplicated()])
    sims[slr] = output[slr].columns
    output[slr] = np.swapaxes(np.array(output[slr]),0,1)
    nsim = output[slr].shape[0]

print('running building LPIs')
building_LPIs = {}

y_t = list(bldgs['Latitude'])
x_t = list(bldgs['Longitude'])
for slr in slrs:
    print('running ',slr)
    X = np.array(points[slr]['lon'])
    Y = np.array(points[slr]['lat'])
    Z = output[slr].T
    z_t = NNR(np.column_stack((x_t, y_t)), np.column_stack((X, Y)), Z, sample_size=-1, n_neighbors=4, weight='distance2')
    building_LPIs[slr] = z_t.T
del points, output
print('running building PGAs')

pga = pd.read_csv(outdir + 'newvs30_pgas_pypsha.csv',header=None)
locs = pd.read_csv(outdir + 'Alameda_new_vs30.csv', index_col=0)
X = np.array(locs['lon'])
Y = np.array(locs['lat'])
Z = np.array(np.array(pga.T))
del pga, locs
z_t = NNR(np.column_stack((x_t, y_t)), np.column_stack((X, Y)), Z, sample_size=-1, n_neighbors=4, weight='distance2')
building_PGAs = z_t.T

LR_gs = eq_shaking_loss(bldgs, building_PGAs)

LR = {}
for slr in building_LPIs.keys():
    LR_liq = liq_frag_func(building_LPIs[slr])
    LR_liq = np.nan_to_num(LR_liq) #replace NaN with 0
    LR[slr] = np.maximum(LR_gs[sims[slr],:],LR_liq)

print('calculating losses')

bldg_losses = {}
reg_losses = {}
for slr in LR.keys():
    LR_both = LR[slr].clip(max=1)
    bldg_losses[slr] = []
    for sim in range(LR_both.shape[0]):
        temp = LR_both[sim,:] * bldgs['ImprovementValue']
        bldg_losses[slr].append(temp)
    bldg_losses[slr] = np.array(bldg_losses[slr])
    reg_losses[slr] = sum(bldg_losses[slr].T)

eq_bldgs = bldgs.copy()
with open ('/scratch/groups/bakerjw/emongold/eq_bldg_loss_nsi_hazus.pkl','wb') as f:  # 
    pickle.dump(bldg_losses,f)

print('saved pickle')

lambda_M = get_pM_pypsha(outdir+'event_save_newvs30.pickle')
bldg_eal = {}
for slr in reg_losses.keys():
    bldg_eal[slr] = lambda_M * bldg_losses[slr].T
    eq_bldgs['EAL_'+ str(slr)] = np.sum(bldg_eal[slr],1)
    eq_bldgs['EAL_norm_' + str(slr)] = eq_bldgs['EAL_'+ str(slr)]/eq_bldgs['ImprovementValue']
eq_bldgs['geometry'] = eq_bldgs['geometry'].apply(lambda x: shapely.wkt.loads(x))
eq_bldgs = gpd.GeoDataFrame(eq_bldgs, crs="EPSG:4326")

eq_bldgs.to_csv('/scratch/groups/bakerjw/emongold/eq_bldgs_out_nsi_hazus.csv')
print('finished')
