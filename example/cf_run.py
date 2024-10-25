## script cf_run.py to run the coastal flooding model on Alameda
## created by Emily Mongold
## last updated 08/08/2024

import pandas as pd
import numpy as np
import geopandas as gpd
import copy
import utm
import pickle

from regional_mh_risk.coastal_flooding import wing_depth_damage_array, assign_flood_depth, make_ART_table, interpolate_log_linear, create_mask_array
from regional_mh_risk.preprocess import setup_grid

cf_folder = './cropped_coastal_flood_rasters/'
bldgs = pd.read_csv('./bldgs_filtered_nsi.csv')

print('finished imports')

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

print('setup buildings')

modified_lookup_dict = {}

for i, row in bld_mayflood.iterrows():
    building_id = row.name
    # Create a copy of the lookup table for this building
    modified_lookup = lookup_table.copy()
    
    # Iterate through the modified lookup table to replace 'inun##' with actual values
    for col in modified_lookup.columns:
        for idx, value in modified_lookup[col].items():
            if pd.notnull(value):  # If the cell is not NaN
                # Replace the placeholder with the actual value from bld_mayflood
                inun_col = value  # Extract the column name that matches the placeholder
                if inun_col in row:
                    modified_lookup.at[idx, col] = row[inun_col]
    
    # Store the modified lookup table in the dictionary with the building ID as the key
    modified_lookup_dict[building_id] = modified_lookup
empty_lookup_dict = copy.deepcopy(modified_lookup_dict)

filled_lookup_dict = {}
for key, df in modified_lookup_dict.items():
    temp_df = copy.deepcopy(df)
    filled_lookup_dict[key] = interpolate_log_linear(df) * create_mask_array(temp_df)

## this line is if you want the one with zeros, but modified_lookup_dict has interpolated values
modified_lookup_dict = copy.deepcopy(filled_lookup_dict)

# take the average of 36" and 42" as 39" == 1m
for bld_idx in modified_lookup_dict:
    modified_lookup_dict[bld_idx].loc[39] = modified_lookup_dict[bld_idx].loc[[36,42]].mean()
    modified_lookup_dict[bld_idx] = modified_lookup_dict[bld_idx].drop([36,42])
    modified_lookup_dict[bld_idx] = modified_lookup_dict[bld_idx].sort_index()

print('created lookup table')

# apply wing_depth_damage_array to all depths in the bldgs dataframe
bldg_LRs = {}
bldg_loss = {}
nsamples = 100
for bld_idx in modified_lookup_dict:
    # apply depth-damage function to the buildings where index is bld_idx
    bldg_LRs[bld_idx], bldg_loss[bld_idx] = wing_depth_damage_array(modified_lookup_dict[bld_idx],bldgs.loc[bld_idx]['ImprovementValue'],nsamples)
Loss_total = pd.concat(bldg_loss.values(), axis=0).groupby(level=0).sum()

losses = np.arange(0, 2e9, 1e5) # total portfolio is 7e9
exceedance = pd.DataFrame()
lambdas = 1/np.array(modified_lookup_dict[next(iter(modified_lookup_dict))].columns)
# calculate ocurrence rate as the difference between the exceedance values
ocurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
ocurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
ocurrence[-1] = lambdas[-1]
print(sum(ocurrence))

for slr in modified_lookup_dict[next(iter(modified_lookup_dict))].index:
    temp = []
    for loss in losses:
        rate = Loss_total.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (ocurrence) * (1/nsamples)
        temp.append(sum(rate))
    exceedance[slr] = temp

print('losses done, starting EAL calcs')

temp000 = []
temp075 = []
temp150 = []
temp025 = []
temp050 = []
temp100 = []
temp125 = []
indices = []
for bld in bldg_loss.keys():  ## bldg_loss_hazus includes losses for zero depths
    bldloss = (bldg_loss[bld].values*ocurrence).sum(axis = 1)
    temp000.append(bldloss[0].mean()) # where slr is 0.0m
    temp025.append(bldloss[1].mean()) # where slr is 0.25m
    temp050.append(bldloss[2].mean()) # where slr is 0.5m
    temp075.append(bldloss[3].mean()) # where slr is 0.75m
    temp100.append(bldloss[4].mean()) # where slr is 1.0m
    temp125.append(bldloss[5].mean()) # where slr is 1.25m
    temp150.append(bldloss[6].mean()) # where slr is 1.5m
    indices.append(bld)
# fill where bldgs['id'] is the indices with temp
real_indices = []
for i in indices:
    real_indices.append(np.where(bldgs.index == i)[0][0])
floodeal000 = np.zeros(len(bld_gdf))
floodeal025 = np.zeros(len(bld_gdf))
floodeal050 = np.zeros(len(bld_gdf))
floodeal075 = np.zeros(len(bld_gdf))
floodeal100 = np.zeros(len(bld_gdf))
floodeal125 = np.zeros(len(bld_gdf))
floodeal150 = np.zeros(len(bld_gdf))

floodeal000[real_indices] = temp000
floodeal025[real_indices] = temp025
floodeal050[real_indices] = temp050
floodeal075[real_indices] = temp075
floodeal100[real_indices] = temp100
floodeal125[real_indices] = temp125
floodeal150[real_indices] = temp150
bld_gdf['flood_eal_00'] = floodeal000
bld_gdf['flood_eal_25'] = floodeal025
bld_gdf['flood_eal_50'] = floodeal050
bld_gdf['flood_eal_75'] = floodeal075
bld_gdf['flood_eal_100'] = floodeal100
bld_gdf['flood_eal_125'] = floodeal125
bld_gdf['flood_eal_15'] = floodeal150

bld_gdf['E_LR_0.0'] = bld_gdf['flood_eal_00']/bld_gdf['ImprovementValue']
bld_gdf['E_LR_0.25'] = bld_gdf['flood_eal_25']/bld_gdf['ImprovementValue']
bld_gdf['E_LR_0.5'] = bld_gdf['flood_eal_50']/bld_gdf['ImprovementValue']
bld_gdf['E_LR_0.75'] = bld_gdf['flood_eal_75']/bld_gdf['ImprovementValue']
bld_gdf['E_LR_1.0'] = bld_gdf['flood_eal_100']/bld_gdf['ImprovementValue']
bld_gdf['E_LR_1.25'] = bld_gdf['flood_eal_125']/bld_gdf['ImprovementValue']
bld_gdf['E_LR_1.5'] = bld_gdf['flood_eal_15']/bld_gdf['ImprovementValue']

print('completed loss calcs')

# save bldg loss to pickle file
with open('flood_bldg_loss_new.pkl', 'wb') as f:
    pickle.dump(bldg_loss, f)

# save bld_gdf to csv
bld_gdf.to_csv('flood_bldg_outs_new.csv')

print('complete, saved')

