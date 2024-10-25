## script tsu_run.py to run tsunami simulaitons for Alameda
## By Emily Mongold 08/08/2024

import numpy as np
import pandas as pd
import geopandas as gpd
import copy
import pickle

from regional_mh_risk.tsunami import assign_tsunami_depth, suppasri_2013_tff, tsunami_DS_to_loss, assign_elevation, interpolate_log_linear_tsunami
from regional_mh_risk.coastal_flooding import get_tiff_values_flood

bldgs = pd.read_csv('./bldgs_filtered_nsi.csv')
out_tif_folder = './Alameda_tsunami_tifs/'
print('finished imports')

tsunami_bldgs = assign_tsunami_depth(bldgs, out_tif_folder)
tsunami_bldgs = gpd.GeoDataFrame(tsunami_bldgs, geometry=gpd.points_from_xy(tsunami_bldgs.Longitude, tsunami_bldgs.Latitude),crs='EPSG:4326')

out_tif_path = './Alameda_elevation.tif'
tsunami_bldgs = assign_elevation(tsunami_bldgs, out_tif_path)

RPs = ['00072','00100','00200','00475','00975','02475','03000']
SLRs = [0,0.25,0.5,0.75,1,1.25,1.5]
bld_mayflood = tsunami_bldgs[tsunami_bldgs['Amplitude_03000'] + 1.5 - tsunami_bldgs['elevation'] > 0]  # basing flood possibility off of tsunami amplitude

print('loaded data')

modified_lookup_dict = {}
for i, row in bld_mayflood.iterrows():
    building_id = row.name
    # Create a copy of the lookup table for this building
    modified_lookup = pd.DataFrame(columns=RPs, index = SLRs)
    # Fill in with 0m SLR row
    for RP in RPs:
        modified_lookup[RP][0] = row[f'Flowdepth_{RP}'] - row['first_floor_elev']
        for SLR in SLRs[1:]:
            if row[f'Flowdepth_{RP}'] > 0:
            # for each of the next rows, fill in with the value + the row index
                modified_lookup[RP][SLR] = max(row[f'Flowdepth_{RP}'] + SLR - row['first_floor_elev'],0)
            elif (row[f'Amplitude_{RP}'] + SLR - row['elevation'] > -2) &  (row[f'Amplitude_{RP}'] - row['elevation'] < -2):
                # for places where the previous value was zero, only if the amplitude + SLR - elevation > -2 --> these we would have expected to be flooded but were not
                # calculate delta from projection for 0m SLR
                modified_lookup[RP][SLR] = max(0.95*(row[f'Amplitude_{RP}'] - row['elevation'] + SLR) + 1.78  - row['first_floor_elev'],0)
            elif row[f'Amplitude_{RP}'] - row['elevation'] > -2:
                if RP == '03000':
                    modified_lookup[RP][SLR] = 0
                else:
                    modified_lookup[RP][SLR] = np.nan
            else:  # if the previous was zero and amplitude + SLR - elevation < -2
                modified_lookup[RP][SLR] = 0
    # Store the modified lookup table in the dictionary with the building ID as the key
    modified_lookup_dict[building_id] = modified_lookup
empty_lookup_dict = copy.deepcopy(modified_lookup_dict)

for df in modified_lookup_dict.values():
    interpolate_log_linear_tsunami(df)
filled_lookup_dict = copy.deepcopy(modified_lookup_dict)

print('ran hazard')

# apply supprasi_2013_tff to all flow depths in the bldgs dataframe
nsamples=100
tsunami_DSs = {}
tsunami_LRs = {}
tsunami_loss = {}
for bld_idx in modified_lookup_dict:
    # apply depth-damage function to the buildings where index is bld_idx
    tsunami_DSs[bld_idx] = pd.DataFrame(index=modified_lookup_dict[bld_idx].index, columns=modified_lookup_dict[bld_idx].columns)
    tsunami_LRs[bld_idx] = pd.DataFrame(index=modified_lookup_dict[bld_idx].index, columns=modified_lookup_dict[bld_idx].columns)
    tsunami_loss[bld_idx] = pd.DataFrame(index=modified_lookup_dict[bld_idx].index, columns=modified_lookup_dict[bld_idx].columns)
    # for each row and column, apply the depth-damage function
    for SLR in modified_lookup_dict[bld_idx].index:
        for RP in modified_lookup_dict[bld_idx].columns:
            tsunami_DSs[bld_idx].loc[SLR,RP] = suppasri_2013_tff(modified_lookup_dict[bld_idx].loc[SLR,RP], tsunami_bldgs.loc[bld_idx,'Stories'],nsamples)
            tsunami_LRs[bld_idx].loc[SLR,RP],tsunami_loss[bld_idx].loc[SLR,RP]= tsunami_DS_to_loss(tsunami_DSs[bld_idx].loc[SLR,RP],tsunami_bldgs.loc[bld_idx,'ImprovementValue'])

Loss_total = pd.concat(tsunami_loss.values(), axis=0).groupby(level=0).sum() # maintains the nsamples

# calculate the loss exceedence curve using reg_losses and lambda_M as the return rate
losses = np.arange(0, 2e9, 1e5) # total portfolio is 7e9
exceedance = pd.DataFrame()
num_RPs = [72,100,200,475,975,2475,3000]
lambdas = 1/np.array(num_RPs)  ## this is the exceedance rate
ocurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
ocurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
ocurrence[-1] = lambdas[-1]
for slr in SLRs:
    temp = []
    for loss in losses:
        rate = Loss_total.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (ocurrence) * (1/nsamples)
        temp.append(sum(rate))
    exceedance[slr] = temp

for bld_idx in tsunami_loss:
    for slr in SLRs:
        tsunami_bldgs.loc[bld_idx,'EAL_'+str(slr)] = np.sum(tsunami_loss[bld_idx].loc[slr].apply(lambda x:sum(np.array(x))) * (ocurrence) * (1/nsamples))
    
print('ran losses')

## Export tsunami_bldgs to csv and LRs/loss to pickle
tsunami_bldgs.to_csv('tsunami_bldg_outs_nsi.csv')

with open('tsunami_bldg_loss.pkl', 'wb') as f:
    pickle.dump(tsunami_loss, f)
with open('tsunami_bldg_LRs.pkl', 'wb') as f:
    pickle.dump(tsunami_LRs, f)

print('finished,saved')

