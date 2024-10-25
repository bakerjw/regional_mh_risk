## adapt_raise.py to run raise houses adaptation for Alameda
## By Emily Mongold 08/09/2024

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import copy
import utm
import pickle
import contextily as ctx

from regional_mh_risk.coastal_flooding import wing_depth_damage_array, assign_flood_depth, make_ART_table, interpolate_log_linear, create_mask_array
from regional_mh_risk.preprocess import setup_grid
from regional_mh_risk.postprocess import get_pM_pypsha
from regional_mh_risk.tsunami import assign_tsunami_depth, suppasri_2013_tff, tsunami_DS_to_loss, assign_elevation, interpolate_log_linear_tsunami

cf_folder = 'C:\\Users/Emily/Documents/Work/coastal_flood_rasters/'
bldgs = pd.read_csv('./bldgs_filtered_nsi.csv')
bldgs = gpd.GeoDataFrame(bldgs, geometry=gpd.points_from_xy(bldgs.Longitude, bldgs.Latitude),crs='EPSG:4326')

print('imports complete')

#assign elevations and retrofit based on land and building elevation
out_tif_path = 'C://Users/Emily/Documents/Work/Alameda_elevation.tif'
bldgs = assign_elevation(bldgs, out_tif_path)
bldgs['Raised'] = [True if x < (bldgs['elevation']+bldgs['first_floor_elev']).quantile(0.06) else False for x in (bldgs['elevation']+bldgs['first_floor_elev'])]

print('elevations assigned')

plt.figure(figsize=(8,8))
ax=bldgs[bldgs['Raised'] == 1].plot(c='green', markersize=1)
plt.legend(['Buildings to raise'],markerscale=5)
plt.ylim(37.725, 37.79)
plt.xlim(-122.3, -122.22)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, crs='EPSG:4326')  #Positron  #DarkMatter  #Voyager
plt.savefig('figures/figure9b.png', format='png', dpi=1000, bbox_inches='tight')

print('raised map complete')

points = setup_grid(utmX0=558500, utmY0=4172400, utmX1= 570200, utmY1 = 4183800, width=100, geoplot='./alameda_plots/alameda_city.geojson')
slrs = [0,0.25,0.5,0.75,1.0,1.25,1.5]  # SLR measurements in meters to align with CoSMoS/ the earthquake model
lookup_table = make_ART_table()
lookup_table = lookup_table.drop(0, axis=1)
slr = 0
lookup_table.loc[slr]
RP = lookup_table.loc[slr].dropna().index
lambdas = 1/np.array(RP)
bldgs['utmX'], bldgs['utmY'], _, _ = zip(*bldgs.apply(lambda row: utm.from_latlon(row['Latitude'], row['Longitude']), axis=1))
bldgs = assign_flood_depth(bldgs,cf_folder)
bldgs['geometry'] = gpd.points_from_xy(bldgs.Longitude, bldgs.Latitude)
bld_gdf = gpd.GeoDataFrame(bldgs, geometry='geometry',crs='EPSG:4326')
bld_mayflood = bldgs[bldgs['inun96'] > 0]

modified_lookup_dict = {}
for i, row in bld_mayflood.iterrows():
    building_id = row.name
    modified_lookup = lookup_table.copy()
    for col in modified_lookup.columns:
        for idx, value in modified_lookup[col].items():
            if pd.notnull(value):  # If the cell is not NaN
                inun_col = value  # Extract the column name that matches the placeholder
                if inun_col in row:
                    modified_lookup.at[idx, col] = row[inun_col]
        modified_lookup_dict[building_id] = modified_lookup
empty_lookup_dict = copy.deepcopy(modified_lookup_dict)
filled_lookup_dict = {}
for key, df in modified_lookup_dict.items():
    temp_df = copy.deepcopy(df)
    filled_lookup_dict[key] = interpolate_log_linear(df) * create_mask_array(temp_df)

## this line is if you want the one with zeros, but modified_lookup_dict has interpolated values
modified_lookup_dict = copy.deepcopy(filled_lookup_dict)

for bld_idx in modified_lookup_dict:
    modified_lookup_dict[bld_idx].loc[39] = modified_lookup_dict[bld_idx].loc[[36,42]].mean() # take the average of 36" and 42" as 39" == 1m
    modified_lookup_dict[bld_idx] = modified_lookup_dict[bld_idx].drop([36,42])
    modified_lookup_dict[bld_idx] = modified_lookup_dict[bld_idx].sort_index()

print('set up depth table')

# apply wing_depth_damage_array to all depths in the bldgs dataframe
bldg_LRs_1ft = {}
bldg_loss_1ft = {}
bldg_LRs_3ft = {}
bldg_loss_3ft = {}
nsamples = 100
for bld_idx in modified_lookup_dict:
    if bld_gdf['Raised'].loc[bld_idx]:
        height = 1
    else:
        height = 0
    bldg_LRs_1ft[bld_idx], bldg_loss_1ft[bld_idx] = wing_depth_damage_array(modified_lookup_dict[bld_idx] - height,bldgs.loc[bld_idx]['ImprovementValue'],nsamples)
    bldg_LRs_3ft[bld_idx], bldg_loss_3ft[bld_idx] = wing_depth_damage_array(modified_lookup_dict[bld_idx] - height*3,bldgs.loc[bld_idx]['ImprovementValue'],nsamples)
Loss_total_1ft = pd.concat(bldg_loss_1ft.values(), axis=0).groupby(level=0).sum()
Loss_total_3ft = pd.concat(bldg_loss_3ft.values(), axis=0).groupby(level=0).sum()

losses = np.arange(0, 5e9, 5e5) # total portfolio is 7e9
exceedance_coastal_flood_1ft = pd.DataFrame()
exceedance_coastal_flood_3ft = pd.DataFrame()
lambdas = 1/np.array(modified_lookup_dict[next(iter(modified_lookup_dict))].columns)
occurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
occurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
occurrence[-1] = lambdas[-1]
print(lambdas)
print('calculating exceedances')

for slr in modified_lookup_dict[next(iter(modified_lookup_dict))].index:
    temp1 = []
    temp3 = []
    for loss in losses:
        rate = Loss_total_1ft.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp1.append(sum(rate))
        rate = Loss_total_3ft.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp3.append(sum(rate))
    exceedance_coastal_flood_1ft[slr] = temp1
    exceedance_coastal_flood_3ft[slr] = temp3

out_tif_folder = './Alameda_tsunami_tifs/'
tsunami_bldgs = assign_tsunami_depth(bldgs, out_tif_folder)
tsunami_bldgs = gpd.GeoDataFrame(tsunami_bldgs, geometry=gpd.points_from_xy(tsunami_bldgs.Longitude, tsunami_bldgs.Latitude),crs='EPSG:4326')
RPs = ['00072','00100','00200','00475','00975','02475','03000']
SLRs = [0,0.25,0.5,0.75,1,1.25,1.5]
bld_mayinun = tsunami_bldgs[tsunami_bldgs['Amplitude_03000'] + 1.5 - tsunami_bldgs['elevation'] > 0]  # basing flood possibility off of tsunami amplitude

tsunami_lookup_dict = {}
for i, row in bld_mayinun.iterrows():
    building_id = row.name
    # Create a copy of the lookup table for this building
    tsunami_lookup = pd.DataFrame(columns=RPs, index = SLRs)
    # Fill in with 0m SLR row
    for RP in RPs:
        tsunami_lookup[RP][0] = row[f'Flowdepth_{RP}'] - row['first_floor_elev']
        for SLR in SLRs[1:]:
            if row[f'Flowdepth_{RP}'] > 0:
                tsunami_lookup[RP][SLR] = max(row[f'Flowdepth_{RP}'] + SLR - row['first_floor_elev'],0)
            elif (row[f'Amplitude_{RP}'] + SLR - row['elevation'] > -2) &  (row[f'Amplitude_{RP}'] - row['elevation'] < -2):
                tsunami_lookup[RP][SLR] = max(0.95*(row[f'Amplitude_{RP}'] - row['elevation'] + SLR) + 1.78  - row['first_floor_elev'],0)
            elif row[f'Amplitude_{RP}'] - row['elevation'] > -2:
                if RP == '03000':
                    tsunami_lookup[RP][SLR] = 0
                else:
                    tsunami_lookup[RP][SLR] = np.nan
                tsunami_lookup[RP][SLR] = 0
    tsunami_lookup_dict[building_id] = tsunami_lookup
empty_tsunami_dict = copy.deepcopy(tsunami_lookup_dict)
for df in tsunami_lookup_dict.values():
    interpolate_log_linear_tsunami(df)
filled_tsunami_dict = copy.deepcopy(tsunami_lookup_dict)

print('ran hazard')

# apply suppasri_2013_tff to all flow depths in the bldgs dataframe
nsamples=100
tsunami_DSs_1ft = {}
tsunami_LRs_1ft = {}
tsunami_loss_1ft = {}
tsunami_DSs_3ft = {}
tsunami_LRs_3ft = {}
tsunami_loss_3ft = {}
for bld_idx in tsunami_lookup_dict:
    # apply depth-damage function to the buildings where index is bld_idx
    tsunami_DSs_1ft[bld_idx] = pd.DataFrame(index=tsunami_lookup_dict[bld_idx].index, columns=tsunami_lookup_dict[bld_idx].columns)
    tsunami_LRs_1ft[bld_idx] = pd.DataFrame(index=tsunami_lookup_dict[bld_idx].index, columns=tsunami_lookup_dict[bld_idx].columns)
    tsunami_loss_1ft[bld_idx] = pd.DataFrame(index=tsunami_lookup_dict[bld_idx].index, columns=tsunami_lookup_dict[bld_idx].columns)
    tsunami_DSs_3ft[bld_idx] = pd.DataFrame(index=tsunami_lookup_dict[bld_idx].index, columns=tsunami_lookup_dict[bld_idx].columns)
    tsunami_LRs_3ft[bld_idx] = pd.DataFrame(index=tsunami_lookup_dict[bld_idx].index, columns=tsunami_lookup_dict[bld_idx].columns)
    tsunami_loss_3ft[bld_idx] = pd.DataFrame(index=tsunami_lookup_dict[bld_idx].index, columns=tsunami_lookup_dict[bld_idx].columns)
    # for each row and column, apply the depth-damage function
    for SLR in tsunami_lookup_dict[bld_idx].index:
        for RP in tsunami_lookup_dict[bld_idx].columns:
            tsunami_DSs_1ft[bld_idx].loc[SLR,RP] = suppasri_2013_tff(tsunami_lookup_dict[bld_idx].loc[SLR,RP] - 0.3048*bldgs.loc[bld_idx, 'Raised'], tsunami_bldgs.loc[bld_idx,'Stories'],nsamples)
            tsunami_LRs_1ft[bld_idx].loc[SLR,RP],tsunami_loss_1ft[bld_idx].loc[SLR,RP]= tsunami_DS_to_loss(tsunami_DSs_1ft[bld_idx].loc[SLR,RP],tsunami_bldgs.loc[bld_idx,'ImprovementValue'])
            tsunami_DSs_3ft[bld_idx].loc[SLR,RP] = suppasri_2013_tff(tsunami_lookup_dict[bld_idx].loc[SLR,RP] - 3*0.3048*bldgs.loc[bld_idx, 'Raised'], tsunami_bldgs.loc[bld_idx,'Stories'],nsamples)
            tsunami_LRs_3ft[bld_idx].loc[SLR,RP],tsunami_loss_3ft[bld_idx].loc[SLR,RP]= tsunami_DS_to_loss(tsunami_DSs_1ft[bld_idx].loc[SLR,RP],tsunami_bldgs.loc[bld_idx,'ImprovementValue'])

Loss_tsunami_1ft = pd.concat(tsunami_loss_1ft.values(), axis=0).groupby(level=0).sum() # maintains the nsamples
Loss_tsunami_3ft = pd.concat(tsunami_loss_3ft.values(), axis=0).groupby(level=0).sum()

# calculate the loss exceedence curve using reg_losses and lambda_M as the return rate
losses = np.arange(0, 5e9, 5e5) # total portfolio is 7e9
exceedance_tsunami_1ft = pd.DataFrame()
exceedance_tsunami_3ft = pd.DataFrame()
num_RPs = [72,100,200,475,975,2475,3000]
lambdas = 1/np.array(num_RPs)
occurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
occurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
occurrence[-1] = lambdas[-1]
for slr in SLRs:
    temp1 = []
    temp3 = []
    for loss in losses:
        rate = Loss_tsunami_1ft.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp1.append(sum(rate))
        rate = Loss_tsunami_3ft.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp3.append(sum(rate))
    exceedance_tsunami_1ft[slr] = temp1
    exceedance_tsunami_3ft[slr] = temp3

eqoutdir = './ground_motions/new_gms/'
# earthquake results with no change
with open('eq_bldg_loss_nsi_hazus.pkl','rb') as f:
    earthquake_bldg_loss = pickle.load(f)
earthquake_bldgs = pd.read_csv('eq_bldgs_out_nsi_hazus.csv')
lambda_M = get_pM_pypsha(eqoutdir+'event_save_newvs30.pickle')
exceedance_earthquake = pd.DataFrame()
for slr in earthquake_bldg_loss.keys():
    temp = []
    regloss = np.sum(earthquake_bldg_loss[slr], axis=1)
    for loss in losses:
        where = np.where(regloss > loss)
        temp.append(np.sum(lambda_M[where]))
    exceedance_earthquake[slr] = temp

exceedance_total_1ft = pd.DataFrame(columns=exceedance_earthquake.columns, index=losses)
exceedance_total_3ft = pd.DataFrame(columns=exceedance_earthquake.columns, index=losses)
for slr in exceedance_earthquake.columns:
    slr_to_inches = {0.0: 0, 0.25: 12, 0.5: 18, 0.75: 30, 1.0: 39, 1.25: 48, 1.5: 60} # 1.0m can also map to 42 inches
    slr_in = slr_to_inches.get(slr)
    exceedance_total_1ft.loc[:,slr] = (np.array(exceedance_earthquake[slr]) + np.array(exceedance_coastal_flood_1ft[slr_in]) + exceedance_tsunami_1ft[slr]).values
    exceedance_total_3ft.loc[:,slr] = (np.array(exceedance_earthquake[slr]) + np.array(exceedance_coastal_flood_3ft[slr_in]) + exceedance_tsunami_3ft[slr]).values

print('ran loss')

expected_annual_loss_1ft = {}
for slr_scenario in exceedance_total_1ft.columns:
    exceedance_rates = exceedance_total_1ft[slr_scenario].values
    expected_annual_loss_1ft[slr_scenario] = np.trapz(exceedance_rates, losses)

expected_annual_loss_3ft = {}
for slr_scenario in exceedance_total_3ft.columns:
    exceedance_rates = exceedance_total_3ft[slr_scenario].values
    expected_annual_loss_3ft[slr_scenario] = np.trapz(exceedance_rates, losses)

## save the values for mitigation comparison
with open('adaptation_raised.pkl','wb') as f:
    pickle.dump({'raised_1ft': expected_annual_loss_1ft, 'raised_3ft': expected_annual_loss_3ft,
        'exceedance_raised1ft':exceedance_total_1ft,'exceedance_raised3ft':exceedance_total_3ft}, f)

print('completed loss calcs and saved EAL')

