# preprocess.py
# 
# 
''' This is a python script to hold functions for multi-hazard analysis preprocessing.
Author: Emily Mongold
Date: 30 November 2023
'''

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import geopandas as gpd
import utm
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def convert_geotiff(input_file,output_folder):
    '''
    Function to convert geotiff to EPSG:4326
    inputs: input file is the path to the original geotiff file
            output_folder is the path to the folder where the new geotiff should be saved
    no direct outputs, new geotiffs save to output_folder
    '''
    output_file = output_folder + input_file.split('/')[-1][:-4] + '.tif' # give the output file the same name
    target_crs = 'EPSG:4326'
    new_nodata_value = np.nan

    with rasterio.open(input_file) as src:

        # Read the input data
        data = src.read(1, masked=True)  # Assuming the data is in the first band (1-based index)
        data_masked = np.ma.masked_invalid(data)
        old_nodata_value = src.nodata
        data_masked = np.where(data_masked == old_nodata_value, new_nodata_value, data_masked)
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)

        # Create the output dataset with the desired data type and nodata value
        profile = src.profile
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            dtype='float32',  # Set the desired data type for the output dataset (e.g., 'float32')
            nodata=new_nodata_value, 
            driver='GTiff'
        )
        
        # Perform the reprojection and directly write to the output GeoTIFF
        with rasterio.open(output_file, 'w', **profile, BIGTIFF='YES', TILED='YES', COPY_SRC_OVERVIEWS='YES', COMPRESS='DEFLATE') as dst:
            reproject(
                source=data_masked,
                src_transform=src.transform,
                src_crs=src.crs,
                destination=rasterio.band(dst, 1),
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,  # You can choose a different resampling method if needed
                dst_nodata=new_nodata_value  
            )
    return

def convert_all_geotiffs(input_folder,output_folder):
    ''' 
    Function to convert all geotiffs in a folder to EPSG:4326 using the convert_geotiff function
    inputs: input_folder is the path to the folder with the original geotiffs
            output_folder is the path to the folder where the new geotiffs should be saved
    no direct outputs, new geotiffs save to output_folder
    '''

    for filename in filter(lambda x: x[-4:] == '.tif', os.listdir(input_folder)):
        convert_geotiff(input_folder + filename,output_folder)
    
    return

def setup_grid(utmX0=558700, utmY0=4178000, utmX1= 568200, utmY1 = 4183700, width=100, geoplot='Alameda_shape.geojson'):
    '''function to set up a grid of points
    input: utmX0 is the minimum utmX value (default Alameda)
    input: utmY0 is the minimum utmY value (default Alameda)
    input: utmX1 is the maximum utmX value (default Alameda)
    input: utmY1 is the maximum utmY value (default Alameda)
    input: width is the width of the grid in meters (default 100)
    input: geoplot is the geojson of the area of interest (default Alameda)
    output: points is a pandas dataframe with the lat, lon, utmX, and utmY of each point
    '''
    
    ## make a dataframe of points with lat, lon, utmX, utmY
    points = pd.DataFrame(columns=['lat', 'lon', 'utmX', 'utmY'])
    utmX = utmX0
    utmY = utmY0
    while utmX <= utmX1:
        while utmY <= utmY1:
            lat, lon = utm.to_latlon(utmX, utmY, 10, northern=True)
            points = pd.concat([points, pd.DataFrame([[lat, lon, utmX, utmY]], columns=['lat', 'lon', 'utmX', 'utmY'])])
            utmY += width
        utmX += width
        utmY = utmY0

    # cut off points outside of the area of interest using a shapefile
    gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points['lon'], points['lat']))

    gdf.set_crs(epsg=4326, inplace=True)
    Alameda = gpd.read_file(geoplot)
    points = gpd.sjoin(gdf, Alameda)
    points.reset_index(inplace=True, drop=True)

    return points

