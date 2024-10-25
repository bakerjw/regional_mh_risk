# script to run simulations of multi-hazard risk model
# Emily Mongold, 2024

from .simple_liquefaction import soil_stress, bi_lpi, moss_solve_FS, moss_solve_LPI
import rasterio
import os
import geopandas as gpd
import numpy as np

def liqcalc(sim, fun, store, slr, mags, pgas, C_FC, SLR):
    lpi = {}
    if fun[sim] == 0:  # B&I
        soil, table = soil_stress(store[sim], slr)
        lpi[sim] = bi_lpi(soil, mags[sim], pgas, C_FC[sim], sim)

    elif fun[sim] == 1:  # Moss
        soil, table = soil_stress(store[sim], slr)
        depth, FS = moss_solve_FS(soil, mags[sim], pgas, slr, sim)
        lpi[sim] = moss_solve_LPI(depth, FS, table)[SLR[0]]

    else:
        return 'Problem with fun flag'

    return lpi

def get_tiff_value(geotiff_file, lat, lon):
    with rasterio.open(geotiff_file) as src:
        # Convert latitude and longitude to the corresponding pixel coordinates
        row, col = src.index(lon, lat)
        
        # Read the water depth value at the pixel coordinates
        value = src.read(1, window=((row, row+1), (col, col+1)))

    return value[0][0]

def get_coastal_flood_values(geotiff_folder,points):
    ''' Function to load flood depths at each point from the geotiff file
    points is a dataframe with lat and lon columns
    geotiff_folder is the path to the geotiff folder
    output is an updated dataframe with the depths at each point
    '''
    for filename in filter(lambda x: x[-4:] == '.tif', os.listdir(geotiff_folder)):
        if str(filename.split('_')[2]) == 'flddepth':
            slr = filename.split('_')[3][3:]
            RP = filename[-7:-4]
            vals = []
            for index, row in points.iterrows():
                vals.append(get_tiff_value(geotiff_folder + filename, points['lat'][index], points['lon'][index]))
            points['flood_slr'+slr+'_RP'+RP] = vals

        else:
            continue

    cols = [col for col in points.columns if col.startswith('flood_')]
    points[cols] = points[cols].replace(-9999,0)

    return points

def get_groundwater_values(geotiff_folder,points):
    ''' Function to load groundwater depths at each point from the geotiff file
    points is a dataframe with lat and lon columns
    geotiff_folder is the path to the geotiff folder
    output is an updated dataframe with the depths at each point
    '''
    ## the geotiffs are stored with as County_wt_[tide]_noghb_Kh[Kh.]p[.Kh]_slr[slr.]p[.slr]m.tif
    # open each file in the folder
    for filename in filter(lambda x: x[-4:] == '.tif', os.listdir(geotiff_folder)):
        # extract the tide, Kh, and slr values from the filename
        tide = str(filename.split('_')[2])
        slr = float(str(filename[-9]) + '.' + str(filename[-7:-5]))
        Kh = float(str(filename[-16]) + '.' + str(filename[-14]))
        vals = []
        for index, row in points.iterrows():
            vals.append(get_tiff_value(geotiff_folder + filename, points['lat'][index], points['lon'][index]))
        points['slr'+str(slr)+'_Kh'+str(Kh)+'_'+tide] = vals
    return points

def make_part_gdf(dflpi,cpt):
    '''
    Function make_part_gdf edited to work on a subset of the cpt data
    inputs:
    dflpi: dataframe with LPI values
    cpt: dictionary with CPT data
    output:
    out: geodataframe with LPI values and geometry
    '''
    lats = np.zeros(len(cpt))
    lons = np.zeros(len(cpt))
    part_ind = []
    for ind, name in enumerate(cpt.index):
        lats[ind] = cpt.loc[name]['Lat']
        lons[ind] = cpt.loc[name]['Lon']
        part_ind.append(name)
    out = gpd.GeoDataFrame(dflpi.loc[part_ind], geometry=gpd.points_from_xy(lons, lats), crs='EPSG:4326')
    out.index = cpt.index

    return out