# script to plot outputs of the liquefaction postprocessing
# Emily Mongold, 2022

import geoplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import geopandas as gpd
import pandas as pd
import pickle


plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the y tick labels
plt.rc('legend', fontsize=12)  # fontsize of the legend
plt.rcParams["figure.figsize"] = (10, 7)
orig_cmap = plt.cm.GnBu
cols = orig_cmap(np.linspace(1, 0.3, 10))
cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", cols)


def Alameda_baseplot(plot_folder = 'alameda_plots/'):
    boundary_filename = plot_folder + 'Alameda_shape.geojson'
    buildings_filename = plot_folder + 'alameda_buildings.shp'
    boundary_shape = gpd.read_file(boundary_filename)
    building_shape = gpd.read_file(buildings_filename)

    plt.figure(figsize=(14, 9))
    ax = plt.subplot(111)
    geoplot.polyplot(boundary_shape, facecolor="None", edgecolor="black", ax=ax,
                     linewidth=1.0, extent=(-122.35, 37.745, -122.22, 37.80), zorder=3, alpha=1)
    geoplot.polyplot(building_shape, facecolor="grey", edgecolor="None", ax=ax,
                     linewidth=1.0, extent=(-122.35, 37.745, -122.22, 37.80), zorder=3, alpha=0.5)
    return ax

def Alameda_outline(plot_folder = 'alameda_plots/'):
    boundary_filename = plot_folder + 'Alameda_shape.geojson'
    boundary_shape = gpd.read_file(boundary_filename)

    plt.figure(figsize=(14, 9))
    ax = plt.subplot(111)
    boundary_shape.plot(facecolor="None", edgecolor="black", ax=ax, linewidth=1.0, zorder=3, alpha=1)

    return ax

def all_alameda_outline(plot_folder = 'alameda_plots/'):
    boundary_filename = plot_folder + 'alameda_city.geojson'
    boundary_shape = gpd.read_file(boundary_filename)

    plt.figure(figsize=(14, 9))
    ax = plt.subplot(111)
    boundary_shape.plot(facecolor="None", edgecolor="black", ax=ax, linewidth=1.0, zorder=3, alpha=1)

    return ax

def meanLPI(LPIs):
    # input LPIs is the dict of all LPI values for each SLR scenario, each borehole, and each scen/sim
    # input SLR is the name of the scenario (string)
    # output meanLPIs is the expected LPI value for each scenario at each borehole location

    meanLPIs = {}
    for rise in range(len(LPIs)):
        SLR = list(LPIs.keys())[rise]
        meanLPIs[SLR] = {}
        for bh in range(len(LPIs[SLR])):
            key = list(LPIs[SLR].keys())[bh]
            meanLPIs[SLR][key] = np.mean(LPIs[SLR][key], 1)

    return meanLPIs

def get_dists(imdir):
    # function to get a list of the magnitudes of the events from the output of R2D
    # input: imdir is the directory to the output R2D folder
    # output: mags is a list of the magnitudes of each scenario
    with open(imdir + 'SiteIM.json', 'r') as j:
        contents = json.loads(j.read())

    dist = []
    for scen in range(len(contents['Earthquake_MAF'])):
        dist.append(contents['Earthquake_MAF'][scen]['SiteSourceDistance'])
    newdist = np.zeros(shape=(np.shape(dist)[0], np.shape(dist)[1]))
    # newdist = np.zeros(shape=np.shape(dist))
    for scen in range(len(dist)):
        for loc in range(np.shape(dist)[1]):
            newdist[scen, loc] = dist[scen][loc]

    dists = []
    for i in dist:
        dists.append(i[0])
    return dists, newdist

def get_pM_pypsha(event_path):
    '''
    function get_pM_pypsha to get a np array of annual rates of events from output of pypsha
    input the path to the event_set.pickle file
    output numpy array of probability values (pM)
    '''
    with open(event_path,'rb') as handle:
        event_set = pickle.load(handle)
    pM = np.array(event_set.events.metadata['annualized_rate'])
    return pM

def collapse_mags(dflpi):
    out = pd.DataFrame()
    for rise in np.unique(dflpi['slr']):
        for bh in np.unique(dflpi['bh']):
            temp = dflpi[['avg', 'mags']][(dflpi['bh'] == bh) & (dflpi['slr'] == rise)]
            for M in np.unique(dflpi['mags']):
                new = temp[temp['mags'] == M]

                dicttmp = {'avg': new.mean()['avg'], 'mags': new.mean()['mags'], 'bh': bh, 'slr': rise}
                trial = pd.DataFrame(dicttmp, index=[0])
                out = pd.concat([out, trial], ignore_index=True)
    return out

def make_gdf(dflpi, cpt):
    '''
    function make_gdf to turn dflpi into a geodataframe
    inputs: dflpi is the dataframe of LPI values, 
            cpt is the dictionary of boreholes
    output: out is the geodataframe of LPI values with geometry
    '''
    lats = np.zeros(len(dflpi))
    lons = np.zeros(len(dflpi))
    for row in range(len(dflpi)):
        name = dflpi['bh'][row]
        lats[row] = cpt[name]['Lat']
        lons[row] = cpt[name]['Lon']

    out = gpd.GeoDataFrame(dflpi, geometry=gpd.points_from_xy(lons, lats))

    return out

def rem_nan(dflpi, cpt):
    '''
    function rem_nan to remove the boreholes with nan water depths from the dflpi dataframe
    inputs: dflpi is the dataframe of LPI values, 
            cpt is the dictionary of boreholes
    output: dflpi is the dataframe of LPI values without boreholes with nan water depths
    '''
    names = list(cpt.keys())
    for bh in names:
        if np.isnan(cpt[bh]['Water depth']):
            dflpi = dflpi[dflpi['bh'] != bh]
    return dflpi

def probs(lpis):
    '''
    function probs() to return probability of liquefaction given lpi
    input: lpis is the dictionary of LPI values with format lpis[slr][bh][sim]
    output: prob is the dictionary of probabilities of liquefaction
    '''
    prob = {}
    for slr in list(lpis.keys()):
        prob[slr] = {}
        for bh in list(lpis[slr].keys()):
            prob[slr][bh] = list(map(lambda x: 1 / (1 + (np.exp(-(0.218 * x - 3.092)))), lpis[slr][bh]))
    return prob

def restruct3(prob, store):
    '''
    function restruct3 to take probs and make single dataframe with rows as locs and P_L for each simulation
    inputs: prob is the dictionary of probabilities of liquefaction,
            store is the dictionary of borehole data
    output: gdfp is the geodataframe of probabilities of liquefaction with geometry
    '''
    init = list(store.keys())[0]
    vals = np.zeros(shape=(len(store[init]), len(store)))
    names = []
    for ind, sim in enumerate(store):
        lat = []
        lon = []
        names.append('sim'+str(sim))
        for loc in prob[ind][sim]:
            lat.append(store[sim][int(loc)]['Lat'])
            lon.append(store[sim][int(loc)]['Lon'])
            vals[int(loc), ind] = prob[ind][sim][loc]
    dfp = pd.DataFrame(vals, columns=names)
    dfp['lat'] = lat
    dfp['lon'] = lon
    gdfp = gpd.GeoDataFrame(dfp, geometry=gpd.points_from_xy(dfp['lon'], dfp['lat']))

    return gdfp

def get_pliq(output):
    '''
    function get_pliq to return probability of liquefaction given lpi
    input: output is the array of LPI values 
    output: prob is an array of probabilities of liquefaction
    '''
    pliq = np.zeros(shape=output.shape)
    for sim in range(len(output)):
        temp = list(map(lambda x: 1 / (1 + (np.exp(-(0.218 * x - 3.092)))), output[sim]))

        pliq[sim] = temp

    return np.array(pliq)

