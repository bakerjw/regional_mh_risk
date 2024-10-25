# tsunami.py
'''
This is a python script for running regional tsunami hazard and risk analysis.
Author: Emily Mongold
Date: 19 April 2024
'''

# Import the necessary libraries
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
from pyproj import Proj, transform
from rasterio.merge import merge
import rasterio.mask
from scipy import stats
from scipy.interpolate import interp1d


# Function definitions:
def plot_geotiff(file_path):
    '''
    function plot_geotiff to plot the geotiff file on a map with colorbar for the metric in meters
    inputs:
        file_path: string, path to the geotiff file to plot
    '''
    # Open the file
    with rasterio.open(file_path) as src:
        # Read the first band (the one with the data)
        image = src.read(1)
        image = np.where(image < -3e37, 0.0, image)
        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(image, cmap='viridis', extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
        cbar = fig.colorbar(cax, ax=ax, shrink=0.75)
        cbar.set_label(label=file_path.split('_')[-2] + ' [m]') # label the colorbar with the metric from the file name
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plt.show()
    return

def combine_tsunami_tifs(tif_folder1, tif_folder2, out_tif_folder, utmX0=558500, utmY0=4172400, utmX1= 570200, utmY1 = 4183800):
    '''
    function combine_tsunami_tifs to join two tifs that contain different parts of Alameda
    inputs:
        tif_folder1: string, path to the folder containing the first set of tifs
        tif_folder2: string, path to the folder containing the second set of tifs
        out_tif_folder: string, path to the folder to save the combined tifs
        utmX0: int, x coordinate of the bottom left corner of the area to combine, default Alameda
        utmY0: int, y coordinate of the bottom left corner of the area to combine, default Alameda
        utmX1: int, x coordinate of the top right corner of the area to combine, default Alameda
        utmY1: int, y coordinate of the top right corner of the area to combine, default Alameda
    
    saves the output tif to out_tif_folder, no other outputs

    Hard-coded for Alameda area, filenames for San Francisco and South San Francisco, UTM zone 10N, and EPSG:4326 with positive values
    Names 'Alameda' for the outputs, but uses the metric and RP from the input files
    '''
    # set up the bounding box for the output tif, default is Alameda (UTM in 10N)
    in_proj = Proj('epsg:32610')
    out_proj = Proj('epsg:4326')
    lat0, lon0 = transform(in_proj, out_proj, utmX0, utmY0)
    lat1, lon1 = transform(in_proj, out_proj, utmX1, utmY1)
    # Correcting the positive longitude representation
    lon0 = 360 + lon0 if lon0 < 0 else lon0
    lon1 = 360 + lon1 if lon1 < 0 else lon1

    for file1 in os.listdir(tif_folder1):
        if file1.endswith('.tif'):
            ## combine the two tifs to just the area around alameda
            metric = file1.split('_')[-2]
            RP = file1.split('_')[-1]
            file2 = 'PTHA_18_San_Francisco_South_' + metric + '_' + RP # RP contains '.tif'
            src1 = rasterio.open(tif_folder1 + file1)
            src2 = rasterio.open(tif_folder2 + file2)
            # Merge the files
            mosaic, out_trans = merge([src1, src2])
            # Create a new transform and metadata for the merged file
            out_meta = src1.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": 'epsg:4326'  # Output in the same CRS as the input
            })
            # Clip the merged file to the bounding box
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**out_meta) as mem_dst:
                    mem_dst.write(mosaic)
                    geom = [{
                        'type': 'Polygon',
                        'coordinates': [[
                            (lon0, lat0),
                            (lon1, lat0),
                            (lon1, lat1),
                            (lon0, lat1),
                            (lon0, lat0)
                        ]]
                    }]
                    out_image, out_transform = rasterio.mask.mask(mem_dst, geom, crop=True)
                    out_meta.update({
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    # Save the new geotiff
                    output_filename = out_tif_folder + 'Alameda_' + metric + '_' + RP  # RP contains '.tif'
                    with rasterio.open(output_filename, 'w', **out_meta) as final_dst:
                        final_dst.write(out_image)

def get_tiff_values_flood(geotiff_file, lon, lat):
    '''
    function get_tiff_values_flood to get the flood depth values from a geotiff file for a list of coordinates
    input the path to the geotiff file and the list of UTM-X and UTM-Y coordinates
    output the list of flood depth values
    '''
    with rasterio.open(geotiff_file) as src:
        rows, cols = zip(*[src.index(x, y) for x, y in zip(lon, lat)])
        values = src.read(1)[rows, cols]

    return values

def assign_tsunami_depth(points, tsunami_tif_folder):
    ''' function assign_flood_depth to assign coastal flood depth to each location of interest
    input the points dataframe and the path to folder with coastal flood depth geotiffs
    output the points dataframe with updated inundation depths
    '''
    for filename in filter(lambda x: x[-4:] == '.tif', os.listdir(tsunami_tif_folder)):
        metric = str(filename.split('_')[-2])
        RP = str(filename.split('_')[-1][:-4]) # return period without '.tif'
        depths = get_tiff_values_flood(os.path.join(tsunami_tif_folder,filename), points['Longitude'] + 360, points['Latitude'])
        depths = np.where(depths < -3e37, 0.0, depths)
        points[metric + '_' + RP] = depths #(depths - points['first_floor_elev']).clip(lower=0)
    return points

def suppasri_2013_tff(depth,stories,nsamples=1):
    '''
    function suppasri_2013_tff to apply the fragility function from Suppasri et al. (2013) to provided depths [in meters]
    This is a lognormal fragility function based on depth of water alone.
    inputs:
        depths: numpy array, depths in meters
        stories: number of stories in the building
    outputs:
        damage states: numpy array, damage states based on the fragility function
        DS1 is minor, DS2 is moderate, DS3 is major, DS4 is complete and DS5 is collapsed
    '''
    if stories == 1:
        mu = [-1.7268, -0.858, 0.0481, 0.6872, 0.8134]
        sigma = [1.1462, 0.9395, 0.7115, 0.5288, 0.5941]
    elif stories == 2:
        mu = [-2.008, -0.8747, 0.035, 0.777, 0.9461]
        sigma = [ 1.1873, 0.9053, 0.7387, 0.5153, 0.5744]
    else:  # wood >= 3 stories
        mu = [-2.19, -0.8617, 0.1137, 0.7977, 1.2658]
        sigma = [1.3198, 1.224, 0.844, 0.4734, 0.6242]

    cdf = stats.lognorm(s=sigma, scale=np.exp(mu)).cdf(depth)
    if cdf[-1] > cdf[-2]:
        cdf[-1] = cdf[-2]  # this fixes the problem where we get a negative value for small depths at large DS, and will now be 0
    pdf = np.zeros((6,))
    pdf[0] = 1 - cdf[0]
    pdf[1:5] = -np.diff(cdf, axis=0)
    pdf = np.clip(pdf, 0, None)  ## this will prevent negative probabilities which only occur in edge cases
    pdf[5] = min(cdf[-1],1-sum(pdf[0:5]))
    if pdf[5] < 0:
        if pdf[4] > -1*pdf[5]:
            pdf[4] += pdf[5]
        elif pdf[3] > -1*pdf[5]:
            pdf[3] += pdf[5]
        elif pdf[2] > -1*pdf[5]:
            pdf[2] += pdf[5]
        else:
            pdf[1] += pdf[5]
        pdf[5] = 0
    DS = np.random.choice([0, 1, 2, 3, 4, 5],nsamples, p=pdf)
    
    return DS

def assign_damage_states_tsunami(tsunami_bldgs):
    '''
    function assign_damage_states_tsunami to assign damage states to each location of interest
    input the building dataframe
    output the building dataframe with damage states for each RP event
    '''
    # for each column that starts with 'Flowdepth', extract RP and assign damage states
    for col in filter(lambda x: x[:9] == 'Flowdepth', tsunami_bldgs.columns):
        RP = col[-5:]
        for i in range(len(tsunami_bldgs)):
            if tsunami_bldgs.at[i, col] > 0:
                tsunami_bldgs.at[i, 'DS'+RP] = suppasri_2013_tff(tsunami_bldgs.at[i, col],tsunami_bldgs.at[i, 'Stories'])
            else:
                tsunami_bldgs.at[i, 'DS'+RP] = 0
    return tsunami_bldgs

def tsunami_DS_to_loss(DSs,BRV):
    '''
    function tsunami_DS_to_loss to simulate loss from uniform loss ratio and building value for each RP tsunami
    Based on Goda & DeRisi 2017, Probabilistic Tsunami Loss Estimation Methodology: Stochastic Earthquake Scenario Approach
    
    inputs: tsunami_DS: pandas dataframe with DS for each RP and SLR tsunami, for one building
    output: tsunami_LR: pandas dataframe with LR for each RP and SLR tsunami, for one building
    '''
    out_LRs = []
    # Lower and upper limits for each damage state (0,1,2,3,4,5)
    lower_limits = np.array([0.0, 0.0, 0.1, 0.3, 0.5, 1.0])
    upper_limits = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 1.0])
    for ds in DSs:
        # Map the DS values to corresponding lower and upper limits
        ll_mapped = lower_limits[ds]
        ul_mapped = upper_limits[ds]
        out_LRs.append(np.random.uniform(low=ll_mapped, high=ul_mapped))
    out_loss = np.array(out_LRs)*BRV

    return out_LRs, out_loss

def clip_elevation_tif(tif_path, out_tif_path, utmX0=558500, utmY0=4172400, utmX1= 570200, utmY1 = 4183800):
    '''
    function clip_elevation_tif to reduce the size of the DEM tif to just the area around Alameda
    inputs:
        tif_path: string, path to the folder containing the tif
        out_tif_path: string, path to the folder to save the snipped tif
        utmX0: int, x coordinate of the bottom left corner of the area to combine, default Alameda
        utmY0: int, y coordinate of the bottom left corner of the area to combine, default Alameda
        utmX1: int, x coordinate of the top right corner of the area to combine, default Alameda
        utmY1: int, y coordinate of the top right corner of the area to combine, default Alameda
    
    saves the output tif to out_tif_folder, no other outputs

    Hard-coded for Alameda area, UTM zone 10N, and EPSG:4326 with positive values
    '''
    # set up the bounding box for the output tif, default is Alameda (UTM in 10N)
    in_proj = Proj('epsg:32610')
    out_proj = Proj('epsg:4326')
    lat0, lon0 = transform(in_proj, out_proj, utmX0, utmY0)
    lat1, lon1 = transform(in_proj, out_proj, utmX1, utmY1)

    with rasterio.open(tif_path) as src:
        # Clip the file to the bounding box
        out_image, out_transform = rasterio.mask.mask(src, [{
            'type': 'Polygon',
            'coordinates': [[
                (lon0, lat0),
                (lon1, lat0),
                (lon1, lat1),
                (lon0, lat1),
                (lon0, lat0)
            ]]
        }], crop=True)
        out_meta = src.meta
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        # Save the new geotiff
        with rasterio.open(out_tif_path, 'w', **out_meta) as dest:
            dest.write(out_image)

def assign_elevation(bldgs, elev_tif_path):
    ''' 
    function assign_elevation to assign elevations to each building
    input the bldgs dataframe and the path to folder with coastal flood depth geotiffs
    output the bldgs dataframe with elevations
    '''
    elevs = get_tiff_values_flood(elev_tif_path, bldgs['Longitude'], bldgs['Latitude'])
    bldgs['elevation'] = elevs

    return bldgs

def interpolate_log_linear_tsunami(df):
    '''
    function interpolate_log_linear_tsunami to fill in nan values of the flow depth data
    input: df the dataframe with flow depth data
    output: df with nan values filled in using log-linear interpolation
    '''
    for index,row in df.iterrows():
        RPs = row.index
        logRPs = np.array(np.log(1/RPs.astype(int)))
        nan_index = np.where([np.isnan(x) for x in (row.values)])[0]
        non_nan_index = np.where([~np.isnan(x) for x in (row.values)])[0]
        if len(non_nan_index) <= 1:
            # in the case that only 3000-yr return period is filled in as zero, fill in all values with zero
            row[:] = 0
            continue
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

