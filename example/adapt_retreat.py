## adapt_retreat.py to run managed retreat adaptation for Alameda
## By Emily Mongold 08/10/2024

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import copy
import pickle
from regional_mh_risk.postprocess import get_pM_pypsha
import shapely.wkt
import contextily as ctx

# where the results from pypsha are stored for earthquake
eqoutdir = './ground_motions/new_gms/'

print('Loading data...')

# load the outputs from each individual hazard:
# coastal flooding
with open('flood_bldg_loss_new.pkl','rb') as f:
    flood_bldg_loss = pickle.load(f)
flood_bldgs = pd.read_csv('flood_bldg_outs_new.csv')
# earthquake
with open('eq_bldg_loss_nsi_hazus.pkl','rb') as f:
    earthquake_bldg_loss = pickle.load(f)
earthquake_bldgs = pd.read_csv('eq_bldgs_out_nsi_hazus.csv')
# tsunami
with open('tsunami_bldg_loss.pkl','rb') as f:
    tsunami_bldg_loss = pickle.load(f)
with open('tsunami_bldg_LRs.pkl','rb') as f:
    tsunami_bldg_LRs = pickle.load(f)
tsunami_bldgs = pd.read_csv('tsunami_bldg_outs_nsi.csv')

print('running baseline')

########## BASELINE LOSS FROM ORIGINAL RESULTS ###############
## for coastal flood
losses = np.arange(0, 5e9, 5e5) # total portfolio is 7e9

loss_flood_tot = pd.concat(flood_bldg_loss.values(), axis=0).groupby(level=0).sum()
loss_flood_tot.columns = 1/np.array(loss_flood_tot.columns)

nsamples = len(flood_bldg_loss[next(iter(flood_bldg_loss))].iloc[0,0])

exceedance_coastal_flood = pd.DataFrame()
lambdas = 1/np.array(flood_bldg_loss[next(iter(flood_bldg_loss))].columns)
occurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
occurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
occurrence[-1] = lambdas[-1]
for slr in flood_bldg_loss[next(iter(flood_bldg_loss))].index:
    temp = []
    for loss in losses:
        rate = loss_flood_tot.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_coastal_flood[slr] = temp

## for earthquake
lambda_M = get_pM_pypsha(eqoutdir+'event_save_newvs30.pickle')
exceedance_earthquake = pd.DataFrame()
for slr in earthquake_bldg_loss.keys():
    temp = []
    regloss = np.sum(earthquake_bldg_loss[slr], axis=1)
    for loss in losses:
        where = np.where(regloss > loss)
        temp.append(np.sum(lambda_M[where]))
    exceedance_earthquake[slr] = temp

## for tsunami

num_RPs = [72,100,200,475,975,2475,3000]
nsamples = len(tsunami_bldg_loss[next(iter(tsunami_bldg_loss))].iloc[0,0])

loss_tsunami_tot = pd.concat(tsunami_bldg_loss.values(), axis=0).groupby(level=0).sum()
loss_tsunami_tot.columns = 1/np.array(loss_tsunami_tot.columns.astype(int))

exceedance_tsunami = pd.DataFrame()
lambdas = 1/np.array(num_RPs)
occurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
occurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
occurrence[-1] = lambdas[-1]
for slr in loss_tsunami_tot.index:
    temp = []
    for loss in losses:
        rate = loss_tsunami_tot.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_tsunami[slr] = temp


## combine for total loss exceedance
exceedance_total = pd.DataFrame(columns=exceedance_earthquake.columns, index=losses)
for slr in exceedance_earthquake.columns:
    slr_to_inches = {0.0: 0, 0.25: 12, 0.5: 18, 0.75: 30, 1.0: 39, 1.25: 48, 1.5: 60} # 1.0m can also map to 42 inches
    slr_in = slr_to_inches.get(slr)
    exceedance_total.loc[:,slr] = (np.array(exceedance_earthquake[slr]) + np.array(exceedance_coastal_flood[slr_in]) + exceedance_tsunami[slr]).values

tsunami_bldgs.drop(columns=['Unnamed: 0','Units', 'OccupancyClass'], inplace=True)

earthquake_bldgs.keys()
earthquake_bldgs.drop(columns=['Unnamed: 0', 'Units', 'ImprovementValue', 'Stories', 'Latitude', 'Longitude',
       'OccupancyClass', 'geometry'], inplace=True)
flood_bldgs.keys()
flood_bldgs.drop(columns=['Unnamed: 0', 'Units', 'ImprovementValue', 'YearBuilt', 'Stories', 'Latitude', 'Longitude',
       'OccupancyClass', 'geometry', 'utmX', 'utmY'], inplace=True)
all_bldgs = pd.DataFrame()
all_bldgs['geometry'] = tsunami_bldgs['geometry'].apply(shapely.wkt.loads)
all_bldgs_gdf = gpd.GeoDataFrame(all_bldgs, geometry=all_bldgs['geometry'],crs='EPSG:4326')
tsunami_bldgs.loc[:,tsunami_bldgs.columns[tsunami_bldgs.columns.str.startswith('EAL_')]] = tsunami_bldgs.loc[:,tsunami_bldgs.columns[tsunami_bldgs.columns.str.startswith('EAL_')]].fillna(0)

all_bldgs_gdf['total_eal_0m'] = tsunami_bldgs['EAL_0'] + earthquake_bldgs['EAL_0.0'] + flood_bldgs['flood_eal_00']
all_bldgs_gdf['total_eal_0.75m'] = tsunami_bldgs['EAL_0.75'] + earthquake_bldgs['EAL_0.75'] + flood_bldgs['flood_eal_75']
all_bldgs_gdf['delta_EAL_0.75'] = all_bldgs_gdf['total_eal_0.75m'] - all_bldgs_gdf['total_eal_0m']

n_relocate = int(0.02 * len(all_bldgs_gdf))
print('reloacte', n_relocate, 'buildings')

# find inds to relocate the 2% with highest total_eal_0m
present_norm_inds = (all_bldgs_gdf['total_eal_0m']/tsunami_bldgs['ImprovementValue']).sort_values(ascending=False).index[:int(n_relocate)]
slr_inds = all_bldgs_gdf['delta_EAL_0.75'].sort_values(ascending=False).index[:int(n_relocate)]

all_bldgs_gdf['remove_slr'] = 0
all_bldgs_gdf['remove_present_norm'] = 0

all_bldgs_gdf.loc[slr_inds,'remove_slr'] = 1
all_bldgs_gdf.loc[present_norm_inds,'remove_present_norm'] = 1
all_bldgs_gdf['ImprovementValue'] = tsunami_bldgs['ImprovementValue']
all_bldgs_gdf.to_csv('retreat_bldgs.csv')

# find the overlap where both are removed
all_bldgs_gdf['remove_both'] = all_bldgs_gdf['remove_present_norm'] + all_bldgs_gdf['remove_slr']
all_bldgs_gdf['remove_both'] = all_bldgs_gdf['remove_both'].apply(lambda x: 1 if x == 2 else 0)
both_inds = all_bldgs_gdf[all_bldgs_gdf['remove_both'] == 1].index

print('chosen buildings to retreat')

plt.figure(figsize=(8,8))
ax=all_bldgs_gdf.loc[present_norm_inds].plot(legend=True, markersize=1, c='#a6cee3', legend_kwds={'label': "Highest present-day risk"})
all_bldgs_gdf.loc[slr_inds].plot(ax=ax, legend=True,markersize=1, c='#1f78b4',legend_kwds={'label': "Highest increase in risk"})
all_bldgs_gdf.loc[both_inds].plot(ax=ax, legend=True,markersize=1, c='navy',legend_kwds={'label': "Highest increase in risk"})
plt.legend(['Present-day risk','Increase in risk', 'Both'],markerscale=5)
plt.ylim(37.725, 37.79)
plt.xlim(-122.3, -122.22)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, crs='EPSG:4326')  #Positron  #DarkMatter  #Voyager
plt.savefig('figures/figure9a.png', format='png', dpi=1000, bbox_inches='tight')
print('saved location plot')

# set flood_bldg_loss_present to flood_bldg_loss removing present_inds and same for slr
flood_bldg_loss_present = {k: v for k, v in flood_bldg_loss.items() if k not in present_norm_inds}
flood_bldg_loss_slr = {k: v for k, v in flood_bldg_loss.items() if k not in slr_inds}
# same for tsunami
tsunami_bldg_loss_present = {k: v for k, v in tsunami_bldg_loss.items() if k not in present_norm_inds}
tsunami_bldg_loss_slr = {k: v for k, v in tsunami_bldg_loss.items() if k not in slr_inds}

######## RUN ALL HAZARDS FOR RETREATING BASED ON PRESENT-DAY RISK #########
## for coastal flood
losses = np.arange(0, 5e9, 5e5) # total portfolio is 7e9
loss_flood_tot_present = pd.concat(flood_bldg_loss_present.values(), axis=0).groupby(level=0).sum()
loss_flood_tot_present.columns = 1/np.array(loss_flood_tot_present.columns)
nsamples = len(flood_bldg_loss_present[next(iter(flood_bldg_loss_present))].iloc[0,0])
exceedance_coastal_flood_present = pd.DataFrame()
lambdas = 1/np.array(flood_bldg_loss_present[next(iter(flood_bldg_loss_present))].columns)
for slr in flood_bldg_loss_present[next(iter(flood_bldg_loss_present))].index:
    temp = []
    for loss in losses:
        rate = loss_flood_tot_present.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (lambdas) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_coastal_flood_present[slr] = temp
## for earthquake
lambda_M = get_pM_pypsha(eqoutdir+'event_save_newvs30.pickle')
exceedance_earthquake_present = pd.DataFrame()
earthquake_bldg_loss_present = copy.deepcopy(earthquake_bldg_loss)
for slr in earthquake_bldg_loss_present.keys():
    temp = []
    earthquake_bldg_loss_present[slr][:,present_norm_inds] = 0
    regloss = np.sum(earthquake_bldg_loss_present[slr], axis=1)
    for loss in losses:
        where = np.where(regloss > loss)
        temp.append(np.sum(lambda_M[where]))
    exceedance_earthquake_present[slr] = temp
## for tsunami
num_RPs = [72,100,200,475,975,2475,3000]
nsamples = len(tsunami_bldg_loss_present[next(iter(tsunami_bldg_loss_present))].iloc[0,0])
loss_tsunami_tot_present = pd.concat(tsunami_bldg_loss_present.values(), axis=0).groupby(level=0).sum()
loss_tsunami_tot_present.columns = 1/np.array(loss_tsunami_tot_present.columns.astype(int))
exceedance_tsunami_present = pd.DataFrame()
lambdas = 1/np.array(num_RPs)
for slr in loss_tsunami_tot_present.index:
    temp = []
    for loss in losses:
        rate = loss_tsunami_tot_present.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (1/np.array(num_RPs)) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_tsunami_present[slr] = temp
exceedance_total_present = pd.DataFrame(columns=exceedance_earthquake_present.columns, index=losses)
for slr in exceedance_earthquake_present.columns:
    slr_to_inches = {0.0: 0, 0.25: 12, 0.5: 18, 0.75: 30, 1.0: 39, 1.25: 48, 1.5: 60} # 1.0m can also map to 42 inches
    slr_in = slr_to_inches.get(slr)
    exceedance_total_present.loc[:,slr] = (np.array(exceedance_earthquake_present[slr]) + np.array(exceedance_coastal_flood_present[slr_in]) + exceedance_tsunami_present[slr]).values

######## RUN ALL HAZARDS FOR REMOVING 0.75m SLR RISK #########
## for coastal flood
losses = np.arange(0, 5e9, 5e5) # total portfolio is 7e9
loss_flood_tot_slr = pd.concat(flood_bldg_loss_slr.values(), axis=0).groupby(level=0).sum()
loss_flood_tot_slr.columns = 1/np.array(loss_flood_tot_slr.columns)
nsamples = len(flood_bldg_loss_slr[next(iter(flood_bldg_loss_slr))].iloc[0,0])
exceedance_coastal_flood_slr = pd.DataFrame()
lambdas = 1/np.array(flood_bldg_loss_slr[next(iter(flood_bldg_loss_slr))].columns)
for slr in flood_bldg_loss_slr[next(iter(flood_bldg_loss_slr))].index:
    temp = []
    for loss in losses:
        rate = loss_flood_tot_slr.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (lambdas) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_coastal_flood_slr[slr] = temp
## for earthquake
lambda_M = get_pM_pypsha(eqoutdir+'event_save_newvs30.pickle')
exceedance_earthquake_slr = pd.DataFrame()
earthquake_bldg_loss_slr = copy.deepcopy(earthquake_bldg_loss)
for slr in earthquake_bldg_loss_slr.keys():
    temp = []
    earthquake_bldg_loss_slr[slr][:,present_norm_inds] = 0
    regloss = np.sum(earthquake_bldg_loss_slr[slr], axis=1)
    for loss in losses:
        where = np.where(regloss > loss)
        temp.append(np.sum(lambda_M[where]))
    exceedance_earthquake_slr[slr] = temp
## for tsunami
num_RPs = [72,100,200,475,975,2475,3000]
nsamples = len(tsunami_bldg_loss_slr[next(iter(tsunami_bldg_loss_slr))].iloc[0,0])
loss_tsunami_tot_slr = pd.concat(tsunami_bldg_loss_slr.values(), axis=0).groupby(level=0).sum()
loss_tsunami_tot_slr.columns = 1/np.array(loss_tsunami_tot_slr.columns.astype(int))
exceedance_tsunami_slr = pd.DataFrame()
lambdas = 1/np.array(num_RPs)
for slr in loss_tsunami_tot_slr.index:
    temp = []
    for loss in losses:
        rate = loss_tsunami_tot_slr.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (1/np.array(num_RPs)) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_tsunami_slr[slr] = temp
exceedance_total_slr = pd.DataFrame(columns=exceedance_earthquake_slr.columns, index=losses)
for slr in exceedance_earthquake_slr.columns:
    slr_to_inches = {0.0: 0, 0.25: 12, 0.5: 18, 0.75: 30, 1.0: 39, 1.25: 48, 1.5: 60} # 1.0m can also map to 42 inches
    slr_in = slr_to_inches.get(slr)
    exceedance_total_slr.loc[:,slr] = (np.array(exceedance_earthquake_slr[slr]) + np.array(exceedance_coastal_flood_slr[slr_in]) + exceedance_tsunami_slr[slr]).values

print('ran retreat scenarios')

expected_annual_loss = {}
for slr_scenario in exceedance_total.columns:
    exceedance_rates = exceedance_total[slr_scenario].values
    expected_annual_loss[slr_scenario] = np.trapz(exceedance_rates, losses)

expected_annual_loss_present = {}
for slr_scenario in exceedance_total_present.columns:
    exceedance_rates = exceedance_total_present[slr_scenario].values
    expected_annual_loss_present[slr_scenario] = np.trapz(exceedance_rates, losses)

expected_annual_loss_slr = {}
for slr_scenario in exceedance_total_slr.columns:
    exceedance_rates = exceedance_total_slr[slr_scenario].values
    expected_annual_loss_slr[slr_scenario] = np.trapz(exceedance_rates, losses)

## save the values for mitigation comparison
with open('adaptation_retreat_norm.pkl','wb') as f:
    pickle.dump({'baseline': expected_annual_loss, 'present_retreat': expected_annual_loss_present, 'slr_retreat': expected_annual_loss_slr,
    'exceedance_baseline':exceedance_total,'exceedance_retreat_pres':exceedance_total_present, 'exceedance_retreat_slr':exceedance_total_slr}, f)

print('saved retreat results')