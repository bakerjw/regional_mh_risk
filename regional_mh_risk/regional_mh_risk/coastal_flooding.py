## Functions to run coastal flooding analysis
## By Emily Mongold, 2023

import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import beta

def ca_depth_damage(stories,basement,depth):
    '''
    function ca_depth_damage to recreate the depth-damage function from HAZUS
    valid for California, depends on HAZUS.csv file
    input the number of stories, basement presence, and the water depth
    returns the loss ratio for each depth provided as a fraction
    '''
    hazus = pd.read_csv('HAZUS.csv')
    hazus = hazus[(hazus['Curve'] == 'San Francisco') & (hazus['Type'] == 'Struct')]  
    # take the curve as just the -8 to 10 columns 
    curve = hazus[(hazus['Stories'] == min(stories,2)) & (hazus['Basement'] == basement)][['-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7','8','9','10']]
    vals = np.array(curve)
    # depths = list(map(lambda x:int(x),np.array(curve.columns)))
    LR = np.interp(depth,curve.columns,vals[0])/100
    
    return LR

def ca_depth_damage_array(stories,basement,depth,zero_zero_flag = True):
    '''
    function ca_depth_damage to recreate the depth-damage function from HAZUS
    valid for California, depends on HAZUS.csv file
    input the number of stories, basement presence, and the water depth, can be arrays, of the same length
    input zero_zero_flag is a binary flag indidcating True if damage is zero when depth is zero, default True.
    Output: loss ratio for each depth provided as a fraction
    '''

    # initialize the output array
    LR = np.zeros_like(depth)
    hazus = pd.read_csv('HAZUS.csv')
    hazus = hazus[(hazus['Curve'] == 'San Francisco') & (hazus['Type'] == 'Struct')]
    curve = hazus[(hazus['Stories'] == stories) & (hazus['Basement'] == basement)][['-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7','8','9','10']]
    if zero_zero_flag:
        inds = np.where(np.array(depth) > 0)
        LR[inds] = np.interp(np.array(depth)[inds].astype(float),curve.columns.astype(int),curve.values[0])/100
    else:
        LR = np.interp(np.array(depth).astype(float),curve.columns.astype(int),curve.values[0])/100
    
    return LR

def wing_depth_damage(depth):
    """
    Function wang_depth_damage estimates loss ratio using a beta distribution,based on Wing et al. 2020
    Takes a random sample from the beta distribution to simulate damage.
    
    Input:
    - depth: float, the water depth in feet, currently supports 1 to 7 feet.
    
    Output:
    - damage_ratio: float, estimated damage ratio (0 to 1), based on a sample from the beta distribution.
    """
    
    # Define alpha and beta parameters for depths of 0 to 7 feet
    alpha_params = {0: 1, 1: 0.42, 2: 0.48, 3: 0.49, 4: 0.53, 5: 0.68, 7: 0.80}
    beta_params = {0: 100, 1: 0.8, 2: 0.65, 3: 0.52, 4: 0.41, 5: 0.42, 7: 0.38}
    depth = float(depth)

    if depth > 7:
        # Parameters for depth > 7, use parameters for depth = 7
        a, b = alpha_params[7], beta_params[7]
    elif depth.is_integer() & (depth != 6):
        # For integer depths between 0 and 7, use the given parameters
        a, b = alpha_params[int(depth)], beta_params[int(depth)]
    else:
        # For non-integer depths between 0 and 7, or 6, interpolate parameters
        lower_depth = int(depth)
        upper_depth = lower_depth + 1
        if upper_depth == 6:
            upper_depth = 7
        elif lower_depth == 6:
            lower_depth = 5
        
        # Interpolation
        a = alpha_params[lower_depth] + (alpha_params[upper_depth] - alpha_params[lower_depth]) * (depth - lower_depth)
        b = beta_params[lower_depth] + (beta_params[upper_depth] - beta_params[lower_depth]) * (depth - lower_depth)
    
    # Take a random sample from the beta distribution for the given depth
    damage_ratio = beta(a, b).rvs()
    
    return damage_ratio

def wing_depth_damage_array(depths, BRV, nsamples):
    """
    Function wing_depth_damage_array estimates loss ratio using a beta distribution, 
    based on Wing et al. 2020. Takes multiple random samples from the beta distribution 
    to simulate damage for each depth in a 7x7 matrix.
    
    Inputs:
    - depths: 2D array-like, the water depths in feet for each location in a 7x7 grid.
    - nsamples: int, the number of samples to generate for each depth.
    
    Output:
    - damage_ratios: 2D list, each sub-list contains nsamples estimated damage ratios (0 to 1) for the corresponding depth.
    """
    # Convert depths to a numpy array to enable vectorized operations
    depths_array = np.array(depths, dtype=float)
    
    # Initialize the damage_ratios 2D list of lists
    damage_ratios = [[[] for _ in range(depths_array.shape[1])] for _ in range(depths_array.shape[0])]
    
    # Define alpha and beta parameters for depths of 1 to 7 feet, skipping 6 feet
    alpha_params = np.array([0.01, 0.42, 0.48, 0.49, 0.53, 0.68, 0.80])
    beta_params = np.array([10, 0.8, 0.65, 0.52, 0.41, 0.42, 0.38])
    
    # Process each element in the depths array
    for i in range(depths_array.shape[0]):
        for j in range(depths_array.shape[1]):
            current_depth = depths_array[i, j]
            if current_depth <= 0:
                damage_ratios[i][j] = [0] * nsamples
            elif current_depth >= 7:
                damage_ratios[i][j] = beta(alpha_params[-1], beta_params[-1]).rvs(size=nsamples).tolist()
            else:  # For depths from 1-7 feet
                interp_alpha = np.interp(current_depth, [0, 1, 2, 3, 4, 5, 7], alpha_params)
                interp_beta = np.interp(current_depth, [0, 1, 2, 3, 4, 5, 7], beta_params)
                damage_ratios[i][j] = beta(interp_alpha, interp_beta).rvs(size=nsamples).tolist()
    LR_df = pd.DataFrame(damage_ratios,columns=depths.columns,index=depths.index)
    loss_df = LR_df.apply(lambda x: [np.array(item)*BRV for item in x])
    return LR_df, loss_df

def get_tiff_value_flood(geotiff_file, lat, lon):
    '''
    function get_tiff_value_flood to get the flood depth value from a geotiff file for a given coordinate (single point)
        For mutliple points, utilize get_tiff_values_flood
    
    input the path to the geotiff file and the latitude and longitude
    output the flood depth value
    '''
    with rasterio.open(geotiff_file) as src:
        # Convert latitude and longitude to the corresponding pixel coordinates
        row, col = src.index(lon, lat)
        value = src.read(1, window=((row, row+1), (col, col+1)))

    return value

def get_tiff_values_flood(geotiff_file, utmX, utmY):
    '''
    function get_tiff_values_flood to get the flood depth values from a geotiff file for a list of coordinates
    input the path to the geotiff file and the list of UTM-X and UTM-Y coordinates
    output the list of flood depth values
    '''
    with rasterio.open(geotiff_file) as src:
        rows, cols = zip(*[src.index(x, y) for x, y in zip(utmX, utmY)])
        values = src.read(1)[rows, cols]

    return values

def assign_flood_depth(points, coastal_flood_folder):
    ''' function assign_flood_depth to assign coastal flood depth to each location of interest
    input the points dataframe and the path to folder with coastal flood depth geotiffs
    output the points dataframe with updated inundation depths
    '''
    for filename in filter(lambda x: x[-4:] == '.tif', os.listdir(coastal_flood_folder)):
        slr = str(filename.split('_')[1])
        depths = get_tiff_values_flood(os.path.join(coastal_flood_folder,filename), points['utmX'], points['utmY'])
        depths = np.where(depths > 3e38, 0.0, depths)
        points.loc[:,['inun'+slr]] = (depths - points['first_floor_elev']).clip(lower=0)
    return points

def make_ART_table():
    '''
    function make_ART_table to create the lookup table that relates maps to scenarios (SLR + storm)
    output: lookup_table dataframe 
    '''
    columns = [0, 1, 2, 5, 10, 25, 50, 100]
    rows = [0, 12, 18, 30, 36, 42, 48, 60]
    lookup_table = pd.DataFrame(index=rows, columns=columns).fillna(np.nan)  # Fill with empty strings for clarity

    lookup_table.at[0, 1] = 'inun12'
    lookup_table.at[12, 0] = 'inun12'

    lookup_table.at[0, 5] = 'inun24'
    lookup_table.at[12, 1] = 'inun24'

    lookup_table.at[36, 0] = 'inun36'
    lookup_table.at[18, 2] = 'inun36'
    lookup_table.at[12, 5] = 'inun36'
    lookup_table.at[0, 50] = 'inun36'

    lookup_table.at[48, 0] = 'inun48'
    lookup_table.at[30, 2] = 'inun48'
    lookup_table.at[18, 10] = 'inun48'
    lookup_table.at[12, 50] = 'inun48'

    lookup_table.at[36, 1] = 'inun52'
    lookup_table.at[30, 5] = 'inun52'
    lookup_table.at[18, 25] = 'inun52'
    lookup_table.at[12, 100] = 'inun52'

    lookup_table.at[48, 2] = 'inun66'
    lookup_table.at[42, 5] = 'inun66'
    lookup_table.at[36, 25] = 'inun66'
    lookup_table.at[30, 50] = 'inun66'

    lookup_table.at[60, 1] = 'inun77'
    lookup_table.at[48, 10] = 'inun77'
    lookup_table.at[42, 50] = 'inun77'
    lookup_table.at[36, 100] = 'inun77'

    lookup_table.at[60, 5] = 'inun84'
    lookup_table.at[48, 50] = 'inun84'
    lookup_table.at[42, 100] = 'inun84'

    lookup_table.at[60, 50] = 'inun96'

    return lookup_table

def interpolate_log_linear(df):
    '''
    function interpolate_log_linear to fill in nan values of the depth data
    input: df the dataframe with depth data
    output: df with nan values filled in using log-linear interpolation
    '''
    for index,row in df.iterrows():
        RPs = row.index
        logRPs = np.array(np.log(1/RPs))
        nan_index = np.where([np.isnan(x) for x in (row.values)])[0]
        non_nan_index = np.where([~np.isnan(x) for x in (row.values)])[0]
        for i in nan_index:
            # find the two closest non nan indices and interpolate from them
            # close = non_nan_index[np.argsort(np.abs(non_nan_index - i))[:2]]
            if i < non_nan_index[0]:
                # case where we need to extrapolate from the lowest two values, without going below 0
                low = logRPs[non_nan_index[0]]; high = logRPs[non_nan_index[1]]
                newval = interp1d([low,high],row.values[[non_nan_index[0],non_nan_index[1]]],fill_value='extrapolate')(logRPs[i])
                row[RPs[i]] = max(0,newval)
            elif i > non_nan_index[-1]:
                # case where we need to extrapolate from the highest two values
                low = logRPs[non_nan_index[-2]]; high = logRPs[non_nan_index[-1]]
                newval = interp1d([low,high],row.values[[non_nan_index[-2],non_nan_index[-1]]],fill_value='extrapolate')(logRPs[i])
                row[RPs[i]] = newval
            else:
                # case where we can use on value below and one above
                above = non_nan_index[np.where(non_nan_index > i)][0]
                below = non_nan_index[np.where(non_nan_index < i)][-1]
                low = logRPs[below]; high = logRPs[above]
                row[RPs[i]] = interp1d([low,high],row.values[[below,above]],fill_value='extrapolate')(logRPs[i])
    return df

def create_mask_array(df):
    '''
    function create_mask_array to fill the upper left corner of the dataframe with zeros
    input: df the dataframe with flood depth data
    output: mask_array with zeros in the upper left corner and ones elsewhere
    This can be multiplied by the interpolated values to create a 'floor' that will drop to zero below known depths
    '''
    mask_array = np.zeros_like(df.values, dtype=int)

    # Iterate over rows to create the mask
    for i in range(df.shape[0]):
        found_positive_in_row = False
        for j in range(df.shape[1]):
            # Check for the first positive number in the row
            if df.iloc[i, j] > 0:
                found_positive_in_row = True
                # If the positive is found in the first column, fill all following rows entirely
                if j == 0:
                    mask_array[i:] = 1
                    return mask_array  # Early exit since all following rows are filled
                break
        
        # Start filling with ones after the first positive number is found
        if found_positive_in_row:
            mask_array[i, j:] = 1

    return mask_array