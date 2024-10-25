## building_inventory.py to generate the housing inventory for Alameda
## By Emily Mongold, 2024-08-09

import pandas as pd
import numpy as np
import geopandas as gpd
from NNR import NNR
import requests
import json

# read inventory from tax assessor
bldgs = pd.read_csv('./Alameda_TA_Inventory.csv')

### IMPORT DATA FROM NATIONAL STRUCTURES INVENTORY API
BASE_URL = 'https://nsi.sec.usace.army.mil/nsiapi/structures?fips=06001&fmt=fc' # https://nsi.sec.usace.army.mil/nsiapi/

# Make the API request
response = requests.get(BASE_URL) 
# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Save the data to a file (optional)
    with open('alameda_structures.json', 'w') as file:
        json.dump(data, file, indent=4)
    print("Data successfully retrieved and saved to 'alameda_structures.json'")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
    print(response.text)

## Filter the data
bldgs['OccupancyClass'] = bldgs['UseCategory'].apply(lambda x: 'RES1' if x in ['Single-family Detached', 'Single-family Attached', 'Multi-Family 2-4', 'Duet Home'] else 'RES3')
bldgs['geometry'] = gpd.points_from_xy(bldgs['Longitude'], bldgs['Latitude'])
bld_gdf = gpd.GeoDataFrame(bldgs, geometry='geometry',crs='EPSG:4326')

# keep only RES1 within alameda city
bld_gdf = bld_gdf[bld_gdf.within(gpd.read_file('./alameda_plots/alameda_city.geojson').iloc[0].geometry)]
bld_gdf = bld_gdf[bld_gdf['OccupancyClass'] == 'RES1']

# remove buildings with more than 3 stories
bld_gdf = bld_gdf[bld_gdf['Stories'] <= 3]

print('interpolating missing info')

## Because some buildings have 0 stories, but appear to be actual separate buildings on the map, I will fill in by 
# nearest-neighbor interpolation of those with building heights and rounding to the nearest integer
zero_story_ind = bld_gdf[bld_gdf['Stories'] == 0].index
non_zero_story_ind = bld_gdf[bld_gdf['Stories'] > 0].index
y_t = list(bld_gdf['Latitude'][zero_story_ind])
x_t = list(bld_gdf['Longitude'][zero_story_ind])
X = list(bld_gdf['Longitude'][non_zero_story_ind])
Y = list(bld_gdf['Latitude'][non_zero_story_ind])
Z = np.array(bld_gdf['Stories'][non_zero_story_ind])
z_t = NNR(np.column_stack((x_t, y_t)), np.column_stack((X, Y)), Z, sample_size=-1, n_neighbors=4, weight='distance2')
# round each z_t to the nearest integer
bld_gdf.loc[zero_story_ind,['Stories']] = np.round(z_t)

## Repeat for buildings with 0 year built
zero_year_ind = bld_gdf[bld_gdf['YearBuilt'] == 0].index
non_zero_year_ind = bld_gdf[bld_gdf['YearBuilt'] > 0].index
y_t = list(bld_gdf['Latitude'][zero_year_ind])
x_t = list(bld_gdf['Longitude'][zero_year_ind])
X = list(bld_gdf['Longitude'][non_zero_year_ind])
Y = list(bld_gdf['Latitude'][non_zero_year_ind])
Z = np.array(bld_gdf['YearBuilt'][non_zero_year_ind])
z_t = NNR(np.column_stack((x_t, y_t)), np.column_stack((X, Y)), Z, sample_size=-1, n_neighbors=4, weight='distance2')
# round each z_t to the nearest integer
bld_gdf.loc[zero_year_ind,['YearBuilt']] = np.round(z_t)

print('assigning elevations from NSI data')

## Assign the first floor elevations
rows = []
# Iterate through the features in the data
for feature in data.get('features', []):
    properties = feature.get('properties', {})
    found_ht = properties.get('found_ht', None)
    x = properties.get('x', None)
    y = properties.get('y', None)
    if properties.get('occtype', None)[0:3] == 'RES':
        # Add the data to the list as a dictionary
        rows.append({'found_ht': found_ht, 'x': x, 'y': y,'occtype': properties.get('occtype', None)})
# Concatenate the list of dictionaries into the DataFrame
nsi_inventory = pd.DataFrame(rows)
# change x and y to longitude and latitude
nsi_inventory.columns = ['found_ht', 'lon', 'lat','occtype']
nsi_gdf = gpd.GeoDataFrame(nsi_inventory, geometry=gpd.points_from_xy(nsi_inventory['lon'],nsi_inventory['lat']),crs='EPSG:4326')
# cut off to keep just those in Alameda city
nsi_gdf = nsi_gdf[nsi_gdf.within(gpd.read_file('./alameda_plots/alameda_city.geojson').iloc[0].geometry)]
# assign the first floor elevation to the buildings in the inventory based on the closest NSI building
# use the NNR function to find the nearest neighbor
y_t = list(bld_gdf['Latitude'])
x_t = list(bld_gdf['Longitude'])
X = list(nsi_gdf['lon'])
Y = list(nsi_gdf['lat'])
Z = np.array(nsi_gdf['found_ht'])
z_t = NNR(np.column_stack((x_t, y_t)), np.column_stack((X, Y)), Z, sample_size=-1, n_neighbors=1, weight='distance2')
bld_gdf['first_floor_elev'] = z_t

bld_gdf.to_csv('./bldgs_filtered_nsi.csv',index=False)

print('saved inventory to bldgs_filtered_nsi.csv')