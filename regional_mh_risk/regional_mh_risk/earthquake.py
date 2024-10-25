# earthquake.py

"""
This is a Python script for running a regional earthquake hazard and risk analysis.
Author: Emily Mongold
Date: 30 November 2023
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import itertools
from pypsha import psha
import pickle
import numpy.matlib as nm
import utm
from .NNR import NNR
import random
from scipy import stats

# Function definitions, many from regional_liquefaction package

def get_pgas_from_grid_pypsha(imdir, nsim, names, geoplot, width = 100, utmX0=558700, utmY0=4178000, shape = [100, 60, 20]):
    '''
    function get_pgas_from_grid_pypsha to get the pgas for each site based on pypsha output
    inputs:
        imdir is the filepath where Alameda_sites.csv and pgas_pypsha.csv are located
        nsim is the number of simulations
        names is the names of the sites
        geoplot is the filepath to the geoplot of the area
        width, utmX0, utmY0, and shape are parameters for the grid where pypsha was run, default Alameda
    outputs:
        pgas is a dictionary with the pgas for each site
    '''
    pgas = {}
    site_file = imdir + "Alameda_sites.csv"
    pgaZ = np.genfromtxt(imdir + 'pgas_pypsha.csv', delimiter=',')
    data = pd.read_csv(site_file)
    lons = data['x']
    lats = data['y']

    nsim = len(pgaZ)  # changed since the shape of pgaZ is nsim x ngridloc
    utmX = np.zeros(len(lons))
    utmY = np.zeros(len(lons))
    for i in range(len(utmX)):
        (utmX[i], utmY[i], reg, northrn) = utm.from_latlon(lats[i], lons[i])

    X = (utmX - utmX0) / width
    Y = (utmY - utmY0) / width

    xs = range(shape[0])
    x = nm.repmat(xs, 1, shape[1])
    x_t = list(x[0])

    y_t = []
    y = []
    for i in range(shape[1]):
        y.append(np.full(shape[0], i))
    for i in range(len(y)):
        for j in list(y[i]):
            y_t.append(j) 
    utmX = list(map(lambda x: (width * x) + utmX0, x_t))
    utmY = list(map(lambda x: (width * x) + utmY0, y_t))

    grid = pd.DataFrame(columns=['y', 'x'])
    grid['y'] = y_t
    grid['x'] = x_t
    LAT = np.zeros(len(utmX))
    LON = np.zeros(len(utmY))
    for i in range(len(utmX)):
        lat, lon = utm.to_latlon(utmX[i], utmY[i], reg, northrn)
        LAT[i] = lat
        LON[i] = lon
    grid['lat'] = LAT
    grid['lon'] = LON

    Alameda = gpd.read_file(geoplot)
    for i in range(len(names)):
        pgas[i] = []
    for sim in range(nsim):
        Z = []
        for i in range(pgaZ.shape[1]):  # pga at each location
            Z.append(pgaZ[sim,i])
        Z = np.array(Z)
        z_t = NNR(np.array([x_t, y_t]).T, np.array([X, Y]).T, Z, sample_size=-1, n_neighbors=4, weight='distance2')
        grid['pga'] = z_t
        gdf = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid['lon'], grid['lat']))
        points = gpd.sjoin(gdf, Alameda)
        points.reset_index(inplace=True, drop=True)
        for i in range(len(names)):
            pgas[i].append(points['pga'][i])

    for i in range(len(names)):
        pgas[i] = np.array(pgas[i])

    return pgas

def get_pgas_for_cpt_pypsha(maindir, nsim, cpts, width = 100, utmX0=558700, utmY0=4178000, shape = [100, 60, 20]):
    ''' function get_pgas_for_cpt_pypsha to get the pgas for each cpt location based on pypsha output
    inputs maindir where out_pypsha folder is and where Alameda_sites.csv is stored
    nsim is the number of simulations
    cpts is the cpt dictionary
    width, utmX0, utmY0, and shape are parameters for the grid where pypsha was run
    outputs pga numpy array with shape (nloc,nsim)
    '''
    
    imdir = maindir + 'out_pypsha/'
    width = 100
    utmX0=558700
    utmY0=4178000
    shape = [100, 60, 20]
    pgas = {}
    site_file = maindir + "Alameda_sites.csv"
    pgaZ = np.genfromtxt(imdir + 'pgas_pypsha.csv', delimiter=',')
    data = pd.read_csv(site_file)
    lons = data['x']
    lats = data['y']

    # nsim = len(pgaZ)  # changed since the shape of pgaZ is nsim x ngridloc
    utmX = np.zeros(len(lons))
    utmY = np.zeros(len(lons))
    for i in range(len(utmX)):
        (utmX[i], utmY[i], reg, northrn) = utm.from_latlon(lats[i], lons[i])

    X = (utmX - utmX0) / width
    Y = (utmY - utmY0) / width
    x_t = []
    y_t = []
    pgas = np.zeros(shape=(len(cpts),nsim))
    for i in cpts:
        x_t.append((cpts[i]['UTM-X'] - utmX0) / width)
        y_t.append((cpts[i]['UTM-Y'] - utmY0) / width)
    for sim in range(nsim):
        Z = []
        for i in range(pgaZ.shape[1]):  # pga at each location
            Z.append(pgaZ[sim,i])
        Z = np.array(Z)

        z_t = NNR(np.array([x_t, y_t]).T, np.array([X, Y]).T, Z, sample_size=-1, n_neighbors=4, weight='distance2')
        pgas[:,sim] = z_t

    return pgas

def get_mags_pypsha(event_path):
    '''function to get a np array of magnitudes of events from output of pypsha
    input the path to the event_set.pickle file
    output numpy array of magnitude values (mags)
    '''
    with open(event_path,'rb') as handle:
        event_set = pickle.load(handle)
    mags = np.array(event_set.events.metadata['magnitude'])

    return mags

def run_pypsha_pgas(imdir,gmm, input_name = 'event_save.pickle', output_name = 'pgas_pypsha.csv'):
    '''function to obtain pgas from pypsha output file based on gmm for each scenario
    inputs: imdir- directory where event_set.pickle is stored, gmm- list of indices of gmm to use [0-2]
    optional: input_name- name of the input file (default 'event_save.pickle'), output_name- name of the output file (default 'pgas_pypsha.csv')
    no outputs- saves a file to imdir called output_name (default 'pgas_pypsha.csv')
    '''
    with open(imdir + input_name,'rb') as handle:
        event_set = pickle.load(handle)
    sa_intensity_ids = [item[:-4] for item in list(event_set.events.intensity_filelist['filename'])]    
    ask_events = event_set.maps[sa_intensity_ids[0]]
    ask_events = ask_events.reset_index(level='map_id')
    ask_events = ask_events[ask_events['map_id'] == 0]
    ask_events.drop('map_id', axis=1,inplace=True)
    
    pga = np.zeros(ask_events.shape)
    n = 0
    for scen in event_set.maps[sa_intensity_ids[0]]['site0'].keys()[:(2*len(gmm))]:  ## The 2*len(gmm) is a hack to get the right number of scenarios for now
        if scen[-1] == 0:
            mod = int(gmm[n])
            pga[n,:] = event_set.maps[sa_intensity_ids[mod]].loc[scen]
            n += 1
            
    np.savetxt(imdir + output_name,pga,delimiter=',')

    return

def run_pypsha(site_file,nmaps,outdir):
    '''
    function run_pypsha to run pypsha and obtain rupture scenarios
    inputs: site_file- file with site information  
        nmaps- number of maps to generate, at least 2
        outdir- directory to save output
    no outputs-- saves to a file called 'event_save.pickle'

    Note: has some fixed parameters: PGA as the IM, attenuations [1,2,4] (ASK,BSSA, CY)
    '''
    # have some fixed parameters, such as attenuations and PGA as IM
    test_site = psha.PSHASite(name = 'site',
                            site_filename = site_file,
                            erf=1, intensity_measures = [1],
                            attenuations = [1,2,4],
                            overwrite=True)
    test_site.write_opensha_input(overwrite = True)
    test_site.run_opensha(overwrite= True, write_output_tofile = True)
    event_set = psha.PshaEventSet(test_site)
    sa_intensity_ids = [item[:-4] for item in list(event_set.events.intensity_filelist['filename'])]
    
    event_set.generate_sa_maps(sa_intensity_ids, nmaps)
    with open(outdir + 'event_save.pickle','wb') as handle:
        pickle.dump(event_set, handle)

    return

def ca_cl(year):
    '''
    Function ca_cl to determine the code level of California buildings based on year of construction
    Input: year- year of construction, based on Hazus Inventory Manual
    Output: cl- code level, out of 'HC' high code, 'MC' medium code, or 'PC' pre-code
    '''
    if year <= 1940:
        cl = 'PC'
    elif year <= 1973:
        cl = 'MC'
    elif year > 1973:
        cl = 'HC'
    else:
        print('Invalid construction year')
        cl = np.nan

    return cl

def type_year(OCC, year):
    ''' Generate a structural type based on year and occupancy class. '''
    n = random.random()
    if OCC == 'RES1':
        return 'W1' if n < 0.99 else 'RM1L'
    
    if OCC == 'RES3':
        if year <= 1950:
            thresholds = [0.73, 0.74, 0.75, 0.76, 0.82, 0.85, 0.88, 0.89, 0.98]
            types = ['W1', 'S1L', 'S2L', 'S3', 'S5L', 'C2L', 'C3L', 'RM1L', 'URML', 'MH']
        elif year <= 1970:
            thresholds = [0.72, 0.73, 0.75, 0.77, 0.78, 0.84, 0.86, 0.94, 0.97]
            types = ['W1', 'S1L', 'S2L', 'S3', 'S5L', 'C2L', 'C3L', 'RM1L', 'URML', 'MH']
        else:
            thresholds = [0.73, 0.75, 0.78, 0.84, 0.85, 0.86, 0.95]
            types = ['W1', 'S3', 'S4L', 'C2L', 'C3L', 'PC2L', 'RM1L', 'MH']
        
        for t, type in zip(thresholds, types):
            if n < t:
                return type
        return 'MH'

def get_theta(STR, cl):
    ''' Get the median of the lognormally distributed fragility function. '''
    theta_values = {
        'PC': {
            'W1': [0.18, 0.29, 0.51, 0.77],
            'S1L': [0.09, 0.13, 0.22, 0.38],
            'S2L': [0.11, 0.14, 0.23, 0.39],
            'S3': [0.08, 0.10, 0.16, 0.30],
            'S5L': [0.11, 0.14, 0.22, 0.37],
            'C2L': [0.11, 0.15, 0.24, 0.42],
            'C3L': [0.10, 0.14, 0.21, 0.35],
            'RM1L': [0.13, 0.16, 0.24, 0.43],
            'URML': [0.13, 0.17, 0.26, 0.37],
            'MH': [0.08, 0.11, 0.18, 0.34]
        },
        'MC': {
            'W1': [0.24, 0.43, 0.91, 1.34],
            'S1L': [0.15, 0.22, 0.42, 0.80],
            'S2L': [0.20, 0.26, 0.46, 0.84],
            'S3': [0.13, 0.19, 0.33, 0.60],
            'S4L': [0.24, 0.39, 0.71, 1.33],
            'S5L': [0.13, 0.17, 0.28, 0.45],
            'C2L': [0.18, 0.30, 0.49, 0.87],
            'C3L': [0.12, 0.17, 0.26, 0.44],
            'PC2L': [0.24, 0.36, 0.69, 1.23],
            'RM1L': [0.22, 0.30, 0.50, 0.85],
            'URML': [0.14, 0.20, 0.26, 0.46],
            'MH': [0.11, 0.18, 0.31, 0.60]
        },
        'HC': {
            'W1': [0.26, 0.55, 1.28, 2.01],
            'S3': [0.15, 0.26, 0.54, 1.00],
            'S4L': [0.24, 0.39, 0.71, 1.33],
            'C2L': [0.24, 0.45, 0.90, 1.55],
            'C3L': [0.12, 0.17, 0.26, 0.44],
            'PC2L': [0.24, 0.36, 0.69, 1.23],
            'RM1L': [0.30, 0.46, 0.93, 1.57],
            'MH': [0.11, 0.18, 0.31, 0.60]
        }
    }

    return theta_values[cl][STR]

def ff_gen(STR, PGA, cl):
    ''' Generate the fragility function of each structural type for each component type '''
    beta = 0.64
    theta_values = get_theta(STR, cl)
    for i, theta in enumerate(theta_values):
        cdf = stats.lognorm(s=beta, scale=theta).cdf(PGA) 
        pdf = np.diff(np.concatenate(([1], cdf, [0]))) 
        DS_str = np.random.choice([0, 1, 2, 3, 4], p=pdf)

    # Calculate non-structural acceleration sensitive damage state
    theta_values, beta = get_theta_beta_acc_nonstr(STR, cl)  # Should return 4 theta values and 1 beta
    for i, theta in enumerate(theta_values):
        cdf = stats.lognorm(s=beta, scale=theta).cdf(PGA)
        pdf = np.diff(np.concatenate(([1], cdf, [0])))
        DS_anstr = np.random.choice([0, 1, 2, 3, 4], p=pdf)

    # Calculate non-structural drift sensitive damage state
    theta_values, beta = get_theta_beta_drift_nonstr(STR, cl)  # Should return 4 theta values and 1 beta
    for i, theta in enumerate(theta_values):
        cdf = stats.lognorm(s=beta, scale=theta).cdf(PGA)
        pdf = np.diff(np.concatenate(([1], cdf, [0])))
        DS_dnonstr = np.random.choice([0, 1, 2, 3, 4], p=pdf)

    return DS_str, DS_anstr, DS_dnonstr

def get_theta_beta_acc_nonstr(STR, cl):
    ''' 
    Get the median spectral acceleration (theta) and log standard deviation (beta) 
    for nonstructural acceleration-sensitive fragility curves.
    '''
    nonstr_values = {
        'HC': {
            'W1': [[0.30, 0.60, 1.20, 2.40], [0.73, 0.68, 0.68, 0.68]],
            'S1L': [[0.30, 0.60, 1.20, 2.40], [0.67, 0.67, 0.68, 0.67]],
            'S2L': [[0.30, 0.60, 1.20, 2.40], [0.67, 0.67, 0.68, 0.67]],
            'S3': [[0.30, 0.60, 1.20, 2.40], [0.68, 0.67, 0.67, 0.67]],
            'S5L': [[0.30, 0.60, 1.20, 2.40], [0.67, 0.66, 0.67, 0.67]],
            'C2L': [[0.30, 0.60, 1.20, 2.40], [0.69, 0.67, 0.66, 0.65]],
            'C3L': [[0.30, 0.60, 1.20, 2.40], [0.66, 0.65, 0.64, 0.63]],
            'RM1L': [[0.30, 0.60, 1.20, 2.40], [0.72, 0.65, 0.67, 0.65]],
            'URML': [[0.30, 0.60, 1.20, 2.40], [0.70, 0.65, 0.65, 0.65]],
            'MH': [[0.30, 0.60, 1.20, 2.40], [0.65, 0.67, 0.67, 0.67]]
        },
        'MC': {
            'W1': [[0.25, 0.50, 1.00, 2.00], [0.73, 0.68, 0.67, 0.64]],
            'S1L': [[0.25, 0.50, 1.00, 2.00], [0.67, 0.66, 0.67, 0.67]],
            'S2L': [[0.25, 0.50, 1.00, 2.00], [0.66, 0.66, 0.68, 0.66]],
            'S3': [[0.25, 0.50, 1.00, 2.00], [0.67, 0.66, 0.65, 0.65]],
            'S5L': [[0.25, 0.50, 1.00, 2.00], [0.65, 0.66, 0.66, 0.66]],
            'C2L': [[0.25, 0.50, 1.00, 2.00], [0.68, 0.66, 0.68, 0.68]],
            'C3L': [[0.25, 0.50, 1.00, 2.00], [0.64, 0.67, 0.68, 0.67]],
            'RM1L': [[0.25, 0.50, 1.00, 2.00], [0.68, 0.67, 0.67, 0.67]],
            'URML': [[0.25, 0.50, 1.00, 2.00], [0.66, 0.65, 0.64, 0.64]],
            'MH': [[0.25, 0.50, 1.00, 2.00], [0.65, 0.67, 0.67, 0.67]]
        },
        'PC': {
            'W1': [[0.20, 0.40, 0.80, 1.60], [0.72, 0.70, 0.67, 0.67]],
            'S1L': [[0.20, 0.40, 0.80, 1.60], [0.66, 0.68, 0.68, 0.68]],
            'S2L': [[0.20, 0.40, 0.80, 1.60], [0.65, 0.68, 0.68, 0.68]],
            'S3': [[0.20, 0.40, 0.80, 1.60], [0.65, 0.68, 0.68, 0.68]],
            'S5L': [[0.20, 0.40, 0.80, 1.60], [0.65, 0.68, 0.68, 0.68]],
            'C2L': [[0.20, 0.40, 0.80, 1.60], [0.65, 0.67, 0.67, 0.67]],
            'C3L': [[0.20, 0.40, 0.80, 1.60], [0.64, 0.67, 0.67, 0.67]],
            'RM1L': [[0.20, 0.40, 0.80, 1.60], [0.64, 0.67, 0.67, 0.67]],
            'URML': [[0.20, 0.40, 0.80, 1.60], [0.69, 0.67, 0.67, 0.67]],
            'MH': [[0.20, 0.40, 0.80, 1.60], [0.65, 0.65, 0.65, 0.65]]
        }
    }
    theta_list=[]
    beta_list=[]
    for struct, code in zip(STR, cl):
        theta, beta = nonstr_values[code][struct]
        theta_list.append(theta)
        beta_list.append(beta)

    return theta_list, beta_list

def get_theta_beta_drift_nonstr(STR, cl):
    ''' 
    Get the median spectral displacement (theta) and log standard deviation (beta) 
    for nonstructural drift-sensitive fragility curves.
    '''
    drift_nonstr_values = {
        'HC': {
            'W1': [[0.50, 1.01, 3.15, 6.30], [0.85, 0.88, 0.88, 0.94]],
            'S1L': [[0.86, 1.73, 5.40, 10.80], [0.81, 0.85, 0.77, 0.77]],
            'S2L': [[0.86, 1.73, 5.40, 10.80], [0.84, 0.90, 0.97, 0.92]],
            'S3': [[0.54, 1.08, 3.38, 6.75], [0.86, 0.88, 0.96, 0.98]],
            'S5L': [[0.26, 4.32, 9.00, 27.00], [0.88, 0.76, 1.05, 1.09]],
            'C2L': [[0.72, 1.44, 4.50, 9.00], [0.87, 0.88, 0.99, 0.99]],
            'C3L': [[0.72, 1.44, 4.50, 9.00], [0.84, 0.88, 0.90, 0.88]],
            'RM1L': [[1.80, 3.60, 11.25, 22.50], [0.87, 0.83, 0.77, 0.89]],
            'URML': [[0.54, 1.08, 3.15, 6.75], [1.21, 1.01, 1.05, 1.06]],
            'MH': [[0.48, 0.96, 3.00, 6.00], [0.96, 1.05, 1.07, 0.93]]
        },
        'MC': {
            'W1': [[0.50, 1.01, 3.15, 6.30], [0.89, 0.91, 0.90, 1.04]],
            'S1L': [[0.86, 1.73, 5.40, 10.80], [0.71, 0.83, 0.79, 0.87]],
            'S2L': [[0.86, 1.73, 5.40, 10.80], [0.93, 0.99, 0.96, 0.92]],
            'S3': [[0.54, 1.08, 3.38, 6.75], [0.93, 0.98, 1.05, 0.94]],
            'S5L': [[0.26, 4.32, 9.00, 27.00], [0.93, 0.75, 1.00, 0.97]],
            'C2L': [[0.72, 1.44, 4.50, 9.00], [0.96, 1.00, 1.06, 0.93]],
            'C3L': [[0.72, 1.44, 4.50, 9.00], [0.93, 0.96, 0.94, 0.88]],
            'RM1L': [[1.80, 3.60, 11.25, 22.50], [0.87, 0.79, 0.93, 0.99]],
            'URML': [[0.54, 1.08, 3.38, 6.75], [0.94, 0.99, 1.05, 1.08]],
            'MH': [[0.48, 0.96, 3.00, 6.00], [0.96, 1.05, 1.07, 0.93]]
        },
        'PC': {
            'W1': [[0.50, 1.01, 3.15, 6.30], [1.07, 0.88, 1.03, 1.14]],
            'S1L': [[2.16, 4.32, 9.00, 18.05], [0.79, 0.79, 0.77, 1.00]],
            'S2L': [[0.86, 1.73, 5.40, 10.80], [1.06, 0.97, 0.96, 1.04]],
            'S3': [[0.54, 1.08, 3.38, 6.75], [1.08, 0.96, 1.14, 1.05]],
            'S5L': [[0.26, 4.32, 9.00, 27.00], [0.93, 0.96, 1.07, 1.09]],
            'C2L': [[0.72, 1.44, 4.50, 9.00], [0.87, 0.96, 1.04, 1.00]],
            'C3L': [[0.72, 1.44, 4.50, 9.00], [0.88, 0.94, 0.99, 0.88]],
            'RM1L': [[0.72, 1.44, 4.50, 9.00], [1.22, 1.05, 1.03, 1.02]],
            'URML': [[0.54, 1.08, 3.15, 6.75], [1.21, 0.96, 1.01, 1.06]],
            'MH': [[0.48, 0.96, 3.00, 6.00], [1.15, 1.09, 1.14, 0.99]]
        }
    }
    theta_list=[]
    beta_list=[]
    for struct, code in zip(STR, cl):
        theta, beta = drift_nonstr_values[code][struct]
        theta_list.append(theta)
        beta_list.append(beta)

    return theta_list, beta_list

def hazus_loss_from_damage(DS):
    ''' function hazus_loss_from_damage to obtain loss ratio from damage state based on Hazus. Valid for RES1
    input: DS - array of damage states, occ - column of occupancy type for each building
    output: LR - array of loss ratios in the same shape as DS
    '''
    shape = DS.shape
    DS = DS.flatten()
    LR = pd.Series(DS).map({0: 0, 1: 0.005, 2: 0.023, 3: 0.11, 4: 0.234}).fillna(0).values
    LR = LR.reshape(shape)
    return LR

def hazus_loss_from_damage_accel_nonstr(DS):
    ''' function hazus_loss_from_damage to obtain loss ratio from damage state based on Hazus. Valid for RES1
    input: DS - array of damage states, for nonstructural acceleration sensitive
    output: LR - array of loss ratios in the same shape as DS
    '''
    shape = DS.shape
    DS = DS.flatten()
    LR = pd.Series(DS).map({0:0, 1:0.005,2:0.027, 3:0.08,4:0.266}).fillna(0).values
    LR = LR.reshape(shape)
    return LR

def hazus_loss_from_damage_drift_nonstr(DS):
    ''' function hazus_loss_from_damage to obtain loss ratio from damage state based on Hazus. Valid for RES1
    input: DS - array of damage states, for nonstructural drift sensitive
    output: LR - array of loss ratios in the same shape as DS
    '''
    shape = DS.shape
    DS = DS.flatten()
    LR = pd.Series(DS).map({0:0,1:0.01,2:0.05,3:0.25,4:0.50}).fillna(0).values
    LR = LR.reshape(shape)
    return LR

def eq_shaking_loss(bldgs, bldPGA):
    ''' function eq_shaking_loss to calculate loss from ground shaking.
    inputs: bldgs - dataframe of buildings with columns 'OccupancyClass' and 'YearBuilt'
            bldPGA - array of building PGAs
    outputs: LRs - array of building losses
    '''
    beta = 0.64
    cl = [ca_cl(year) for year in bldgs['YearBuilt']]
    STR = [type_year(oc, year) for oc, year in zip(bldgs['OccupancyClass'], bldgs['YearBuilt'])]
    theta = np.array([get_theta(s, c) for s, c in zip(STR, cl)])
    pgas = np.swapaxes(np.tile(bldPGA.T, (4, 1, 1)), 0, -1)
    theta1 = np.tile(theta, (len(bldPGA), 1, 1))
    cdf = stats.lognorm(s=beta, scale=theta1).cdf(pgas)
    pdf = np.concatenate((np.ones((cdf.shape[0], cdf.shape[1], 1)), cdf), axis=2) - \
          np.concatenate((cdf, np.zeros((cdf.shape[0], cdf.shape[1], 1))), axis=2)
    index = pd.MultiIndex.from_product([range(s) for s in pdf.shape], names=['sim', 'bld', 'ds'])
    df = pd.DataFrame({'A': pdf.flatten()}, index=index)['A']
    df = df.unstack(level='ds').swaplevel().sort_index()
    states = [0, 1, 2, 3, 4]
    rng = np.random.default_rng(42)
    df_selections = pd.DataFrame(data=rng.multinomial(n=1, pvals=df.to_numpy()), columns=states, index=df.index)
    DS = df_selections.idxmax(axis=1)
    m, n = len(DS.index.levels[0]), len(DS.index.levels[1])
    LR_str = hazus_loss_from_damage(DS.values.reshape(m, n, -1).swapaxes(0, 1)).squeeze()

    theta,beta = get_theta_beta_acc_nonstr(STR, cl)
    theta1 = np.tile(theta, (len(bldPGA), 1, 1))
    cdf = stats.lognorm(s=beta, scale=theta1).cdf(pgas)
    pdf = np.concatenate((np.ones((cdf.shape[0], cdf.shape[1], 1)), cdf), axis=2) - \
            np.concatenate((cdf, np.zeros((cdf.shape[0], cdf.shape[1], 1))), axis=2)
    df = pd.DataFrame({'A': pdf.flatten()}, index=index)['A']
    df = df.unstack(level='ds').swaplevel().sort_index()
    df_selections = pd.DataFrame(data=rng.multinomial(n=1, pvals=df.to_numpy()), columns=states, index=df.index)
    DS = df_selections.idxmax(axis=1)
    LR_anstr = hazus_loss_from_damage_accel_nonstr(DS.values.reshape(m, n, -1).swapaxes(0, 1)).squeeze()

    theta,beta = get_theta_beta_drift_nonstr(STR, cl)
    theta1 = np.tile(theta, (len(bldPGA), 1, 1))
    cdf = stats.lognorm(s=beta, scale=theta1).cdf(pgas)
    pdf = np.concatenate((np.ones((cdf.shape[0], cdf.shape[1], 1)), cdf), axis=2) - \
            np.concatenate((cdf, np.zeros((cdf.shape[0], cdf.shape[1], 1))), axis=2)
    pdf[np.where(pdf<0)] = 0
    df = pd.DataFrame({'A': pdf.flatten()}, index=index)['A']
    df = df.unstack(level='ds').swaplevel().sort_index()
    df_selections = pd.DataFrame(data=rng.multinomial(n=1, pvals=df.to_numpy()), columns=states, index=df.index)
    DS = df_selections.idxmax(axis=1)
    LR_dnonstr = hazus_loss_from_damage_drift_nonstr(DS.values.reshape(m, n, -1).swapaxes(0, 1)).squeeze()

    return LR_str + LR_anstr + LR_dnonstr

def loss_from_ds_whitman(DS):
    ''' function loss_from_ds_whitman to obtain loss ratio from damage state given input DS is a dataframe
    output Loss ratio is returned as a fraction in the same format as the input
    from  whitman 1973 values
    '''
    mapping = {0:0,1:0.003,2:0.05,3:0.3,4:1.0}
    LR = DS.map(mapping)
    
    return LR

def eq_shaking_ELR(bldgs, bldPGA):
    ''' function eq_shaking_ELR to calculate expected loss ratio from ground shaking.
    inputs: bldgs - dataframe of buildings with columns 'OccupancyClass' and 'YearBuilt'
            bldPGA - array of building PGAs
    outputs: LRs - array of building losses
    '''
    beta = 0.64
    cl = [ca_cl(year) for year in bldgs['YearBuilt']]
    STR = [type_year(oc, year) for oc, year in zip(bldgs['OccupancyClass'], bldgs['YearBuilt'])]
    theta = np.array([get_theta(s, c) for s, c in zip(STR, cl)])
    
    pgas = np.swapaxes(np.tile(bldPGA.T, (4, 1, 1)), 0, -1)
    theta1 = np.tile(theta, (len(bldPGA), 1, 1))
    
    cdf = stats.lognorm(s=beta, scale=theta1).cdf(pgas)
    pdf = np.concatenate((np.ones((cdf.shape[0], cdf.shape[1], 1)), cdf), axis=2) - \
          np.concatenate((cdf, np.zeros((cdf.shape[0], cdf.shape[1], 1))), axis=2)
    
    DS_LR = np.array([0,0.005,0.023,0.11,0.234])
    ELR = pdf*DS_LR
    LRst = ELR.sum(axis=2)

    theta,beta = get_theta_beta_acc_nonstr(STR, cl)
    theta1 = np.tile(theta, (len(bldPGA), 1, 1))
    cdf = stats.lognorm(s=beta, scale=theta1).cdf(pgas)
    pdf = np.concatenate((np.ones((cdf.shape[0], cdf.shape[1], 1)), cdf), axis=2) - \
            np.concatenate((cdf, np.zeros((cdf.shape[0], cdf.shape[1], 1))), axis=2)
    DS_LR = np.array([0,0.005,0.027,0.08,0.266])  #{0:0, 1:0.005,2:0.027, 3:0.08,4:0.266}
    ELR = pdf*DS_LR
    LRa = ELR.sum(axis=2)

    theta,beta = get_theta_beta_drift_nonstr(STR, cl)
    theta1 = np.tile(theta, (len(bldPGA), 1, 1))
    cdf = stats.lognorm(s=beta, scale=theta1).cdf(pgas)
    pdf = np.concatenate((np.ones((cdf.shape[0], cdf.shape[1], 1)), cdf), axis=2) - \
            np.concatenate((cdf, np.zeros((cdf.shape[0], cdf.shape[1], 1))), axis=2)
    DS_LR = np.array([0,0.01,0.05,0.25,0.5])  #{0:0,1:0.01,2:0.05,3:0.25,4:0.50}
    ELR = pdf*DS_LR
    LRd = ELR.sum(axis=2)

    return LRst + LRa + LRd