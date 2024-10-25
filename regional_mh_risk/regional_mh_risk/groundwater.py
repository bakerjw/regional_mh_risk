## Functions to use CoSMoS groundwater data for liquefaction analysis
## By Jaelen Sobers, 2023
## edits by Emily Mongold, 2024

# from .base import *
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xarray as xr
import pandas as pd
import itertools
from NNR import NNR

def convert_crs(input_file, target_crs, new_nodata_value, output_file):
    with rasterio.open(input_file) as src:

        # Read the input data
        data = src.read(1, masked=True)  # Considering data is in the 1st band

        # Mask the data to handle both NaNs and the specific nodata value
        data_masked = np.ma.masked_invalid(data)

        # Get the specific nodata value from the source GeoTIFF
        old_nodata_value = src.nodata

        # Set the specific nodata value to a new nodata value (NaN)
        data_masked = np.where(data_masked == old_nodata_value, new_nodata_value, data_masked)
    
        # Reproject the data to the target CRS (EPSG:4326)
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)

        # Create the output dataset with the desired data type and nodata value
        profile = src.profile
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            dtype='float32',  # Set the desired data type for the output dataset ('ComplexInt16' to 'float32')
            nodata=new_nodata_value,  # Set the new nodata value for the output dataset
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
                resampling=Resampling.nearest,  
                dst_nodata=new_nodata_value  # Set the new nodata value for the output dataset
            )
    return

def interp_wd(points, geotiff, xb, yb):
    # Uses NNR to determine water depth at each point based on GeoTIFF

    # Open into an xarray.DataArray
    geotiff = xr.open_rasterio(geotiff)

    # Covert our xarray.DataArray into a xarray.Dataset
    geotiff_ds = geotiff.to_dataset('band')

    # Rename the variable to a more useful name
    geotiff_ds = geotiff_ds.rename({1: 'wd'})
    
    # Limiting the range of the geoTIFF
    tiff = geotiff_ds.where(geotiff_ds.y < yb[1])
    tiff = tiff.where(tiff.y > yb[0])
    tiff = tiff.where(tiff.x < xb[1])
    tiff = tiff.where(tiff.x > xb[0])

    # Get rid of missing values
    tiff = tiff.dropna(dim="y", how="all")
    tiff = tiff.dropna(dim="x", how="all")
    
    # Water Depth Values
    Z1 = np.array(tiff.wd)
    X1 = np.array(tiff.x)
    Y1 = np.array(tiff.y)
    
    # Storage of water depths including ocean
    X = []
    Y = []
    Z = []

    for (x, y) in itertools.product(range(len(X1)), range(len(Y1))):
        if pd.notna(Z1[y, x]): 
            X.append(X1[x])
            Y.append(Y1[y])
            Z.append(Z1[y, x])
            
    # Storage of water depths only on land
    X_land = []
    Y_land = []
    Z_land = []
    
    for (x, y) in itertools.product(range(len(X1)), range(len(Y1))):
        if pd.notna(Z1[y, x]) and Z1[y, x] != -500:
            X_land.append(X1[x])
            Y_land.append(Y1[y])
            Z_land.append(Z1[y, x])
            
    water_depth_all = NNR(np.array([np.array(points['lon']), np.array(points['lat'])]).T, np.array([X, Y]).T,
                np.array(Z), sample_size=-1, n_neighbors=4, weight='distance2') # Includes interpolation to submereged points
    water_depth_land = NNR(np.array([np.array(points['lon']), np.array(points['lat'])]).T, np.array([X_land, Y_land]).T,
                np.array(Z_land), sample_size=-1, n_neighbors=4, weight='distance2') # Only interpolates to points on land
    
    return water_depth_all, water_depth_land

def get_pixel_wd(geotiff, lat, lon):
    # Accesses water depth at a particlar pixel
    with rasterio.open(geotiff) as src:
        # Convert latitude and longitude to the corresponding pixel coordinates
        row, col = src.index(lon, lat)
        
        # Read the water depth value at the pixel coordinates
        water_depth = src.read(1, window=((row, row+1), (col, col+1)))

    return water_depth[0][0]

def refine_wd(wd_geotiff, points, wd_all, wd_land):
    # Changes submerged and emergent water depths to zero and takes note of whether each point is submerged or not
    # Returns a list of calculatted water depths at each point and a list of the submersive status of all points
    status = [] # 0 == not submerged, 1 == emergent, 2 == submerged
    final_wd = []
    for i in range(len(points)):
        lat = points['lat'][i]
        lon = points['lon'][i]
        pixel_wd = get_pixel_wd(wd_geotiff, lat, lon)
        if pixel_wd == -500: # Determined to be submerged from geotiff
            final_wd.append(0)
            status.append(2)
        elif np.isnan(pixel_wd):
            if wd_all[i] < -130: # Value of -130 accounts for case of point being near at least one submerged point and emergent points
                final_wd.append(0)
                status.append(2) # Determined to be submerged from interpolation
            else:
                point_wd = wd_land[i]
                if point_wd < 0: # Determined to be emergent from interpolation
                    final_wd.append(0)
                    status.append(1)
                else: # Determined to be on land from interpolaton
                    final_wd.append(point_wd)
                    status.append(0)
        else: # Determined to be on land from geotiff
            point_wd = wd_land[i]
            if point_wd < 0: # Determined to be emergent from interpolation
                    final_wd.append(0)
                    status.append(1)
            else: # Determined to be on land from interpolaton
                final_wd.append(point_wd)
                status.append(0)
    
    return final_wd, status