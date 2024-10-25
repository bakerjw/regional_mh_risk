# simple_liquefaction.py

''' This is a python script to perform simple point-based liquefaction calculations for regional risk analysis.
Author: Emily Mongold
Date: 3 January 2024

functions that begin with 'bi_' are based on Boulanger & Idriss (2014)
functions that begin with 'moss_' are based on Moss et al. (2006)
'''
import numpy as np
from scipy.stats import norm
import utm
import datetime
import os
import pandas as pd
import geopandas as gpd
import skgstat as skg
import gstatsim as gs
from sklearn.preprocessing import QuantileTransformer
from scipy.spatial.distance import cdist
import rasterio

def get_tiff_value(geotiff_file, lat, lon):
    with rasterio.open(geotiff_file) as src:
        # Convert latitude and longitude to the corresponding pixel coordinates
        row, col = src.index(lon, lat)
        
        # Read the water depth value at the pixel coordinates
        value = src.read(1, window=((row, row+1), (col, col+1)))

    return value[0][0]

def setup_cpt(datadir):
    ''' function setup_cpt to load USGS cpt data from folder and output a dictionary
    input: datadir is the directory path to the cpt data
    output: cpt is a dictionary with an np directory for each borehole set of cpt data
    '''

    d = {}
    names = []
    cpt = {}
    # Constants
    g = 9.81
    Pa = 0.101325  # MPa
    rho_w = 1  # Mg/m^3
    gamma_w = rho_w * g / 1000  # MPa

    for filename in filter(lambda x: x[-4:] == '.txt', os.listdir(datadir)):

        with open(os.path.join(datadir, filename)) as f:
            name = datadir + filename
            df_temp = pd.read_csv(name, delimiter="\s+", skiprows=17)
            df_temp = df_temp.dropna(axis='columns', how='all')
            df_temp.columns = ['Depth', 'Tip_Resistance', 'Sleeve_Friction', 'Inclination', 'Swave_travel_time']
            df_temp = df_temp[-((df_temp['Sleeve_Friction'] < 0) | (df_temp['Tip_Resistance'] < 0))]

            df_temp = df_temp[df_temp['Depth'] <= 20]
            df_temp['Sleeve_Friction'] = df_temp['Sleeve_Friction'] / 1000  # convert to units of MPa

            temp = pd.DataFrame(np.zeros(shape=(len(df_temp), 7)),
                                columns=['start', 'q_c', 'f_s', 'd', 'dz', 'gamma', 'R_f'])
            temp['q_c'][0] = df_temp['Tip_Resistance'][0] / 2
            temp['f_s'][0] = df_temp['Sleeve_Friction'][0] / 2
            temp['d'][0] = np.average([temp['start'][0], df_temp['Depth'][0]])
            temp['dz'][0] = df_temp['Depth'][0] - temp['start'][0]
            temp['R_f'][0] = 100 * temp['f_s'][0] / temp['q_c'][0]
            temp['gamma'][0] = gamma_w * (0.27 * (np.log10(temp['R_f'][0])) +
                                          0.36 * np.log10(temp['q_c'][0] / Pa) + 1.236)
            for i in range(1, len(df_temp)):
                temp['start'][i] = df_temp['Depth'].iloc[i - 1]
                temp['f_s'][i] = np.average([df_temp['Sleeve_Friction'].iloc[i], df_temp['Sleeve_Friction'].iloc[i - 1]])
                temp['q_c'][i] = np.average([df_temp['Tip_Resistance'].iloc[i], df_temp['Tip_Resistance'].iloc[i - 1]])
                temp['d'][i] = np.average([temp['start'][i], df_temp['Depth'].iloc[i]])
                temp['dz'][i] = df_temp['Depth'].iloc[i] - temp['start'][i]
                temp['R_f'][i] = 100 * temp['f_s'][i] / temp['q_c'][i]
                # Calculating soil unit weight from Robertson and Cabal (2010)
                if temp['R_f'][i] == 0:
                    if temp['q_c'][i] == 0:
                        temp['gamma'][i] = gamma_w * 1.236
                    else:
                        temp['gamma'][i] = gamma_w * (0.36 * np.log10(temp['q_c'][i] / Pa) + 1.236)
                elif temp['q_c'][i] == 0:
                    temp['gamma'][i] = gamma_w * (0.27 * (np.log10(temp['R_f'][i])) + 1.236)
                else:
                    temp['gamma'][i] = gamma_w * (0.27 * np.log10(temp['R_f'][i]) +
                                                  0.36 * np.log10(temp['q_c'][i] / Pa) + 1.236)

            temp['dsig_v'] = temp['dz'] * temp['gamma']

            key = list(dict(l.strip().rsplit(maxsplit=1) for l in open(name) if any(l.strip().startswith(i) for i in 'File name:')).values())[0]
            names.append(key)
            d[key] = dict(l.strip().rsplit('\t', maxsplit=1) for l in open(name) \
                          if (any(l.strip().startswith(i) for i in ('"UTM-X', '"UTM-Y', '"Elev', '"Water depth', 'Date')) and len(l.strip().rsplit('\t', maxsplit=1)) == 2))

            cpt[key] = {}
            cpt[key]['CPT_data'] = temp

            for i in d[key]:
                if i.startswith('"UTM-X'):
                    cpt[key]['UTM-X'] = int(d[key][i])
                elif i.startswith('"UTM-Y'):
                    cpt[key]['UTM-Y'] = int(d[key][i])
                elif i.startswith('"Elev'):
                    cpt[key]['Elev'] = float(d[key][i])
                elif i.startswith('"Water depth'):
                    cpt[key]['Water depth'] = float(d[key][i])
                elif i.startswith('Date'):
                    cpt[key]['Date'] = datetime.datetime.strptime(d[key][i], '%m/%d/%Y')

            if 'Elev' not in cpt[key]:
                cpt[key]['Elev'] = np.nan
            if 'Water depth' not in cpt[key]:
                cpt[key]['Water depth'] = np.nan

    for i in range(len(names)):
        cpt[names[i]]['Lat'], cpt[names[i]]['Lon'] = utm.to_latlon(cpt[names[i]]['UTM-X'], cpt[names[i]]['UTM-Y'], 10,
                                                                   northern=True)
    return cpt

def assign_gwt(cpt, gwt_folder):
    ''' function assign_gwt to assign groundwater depth to each borehole of cpt data
    input the cpt dataframe and the path to folder with groundwater depth geotiffs
    output the cpt dataframe with updated groundwater depth
    '''
    
    for key in cpt:
        cpt[key]['wd'] = pd.DataFrame(columns = ['slr','Kh','tide','gwt'])
        ## loop through the geotiff files in the groundwater depth folder
        for filename in filter(lambda x: x[-4:] == '.tif', os.listdir(gwt_folder)):
            ## extract the tide, Kh, and slr values from the filename
            tide = str(filename.split('_')[2])
            slr = float(str(filename[-9]) + '.' + str(filename[-7:-5]))
            Kh = float(str(filename[-16]) + '.' + str(filename[-14]))
            Kh = np.where(np.array(Kh) == 0.0,10.0,np.array(Kh))
            cpt[key]['wd'] = pd.concat([cpt[key]['wd'],pd.DataFrame([[slr,Kh,tide,get_tiff_value(gwt_folder + filename, cpt[key]['Lat'], cpt[key]['Lon'])]],columns=['slr','Kh','tide','gwt'])])
    return cpt

def liqcalc(sim, fun, store, slr, mags, pgas, C_FC, SLR):
    ''' function liqcalc to run liquefaction calculations for a given simulation
    Inputs are sim (simulation number), fun (liquefaction function), store (simulation soil/cpt data), slr (sea level rise [float]), mags (magnitudes), pgas (PGA values), C_FC (fines content constant), SLR (sea level rise [string])
    Output is lpi (liquefaction potential index)
    '''

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

def bi_find_CSR75(d, M, pga, sig_v, sig_prime_v, q_c1Ncs):
    ''' function bi_find_CSR75 to calculate CSR_75 values from Boulanger and Idriss (2014)
    Inputs are d (depth), M (magnitude), pga (peak ground acceleration), sig_v (vertical stress), sig_prime_v (effective vertical stress), q_c1Ncs (normalized cone tip resistance)
    Output is CSR_75 (cyclic stress ratio for a magnitude 7.5 earthquake)
    '''
    alpha = -1.012 - 1.126 * np.sin((d / 11.73) + 5.133)
    beta = 0.106 + 0.118 * np.sin((d / 11.28) + 5.142)
    r_d = np.exp(alpha + beta * M)
    CSR = 0.65 * (sig_v / sig_prime_v) * pga * r_d
    MSF = 6.9 * np.exp(- M / 4) - 0.058
    Pa = 0.101325  # MPa
    C_sig = np.minimum(1 / (37.3 - 8.27 * q_c1Ncs ** 0.264),0.3)
    K_sig = np.minimum(1 - C_sig * np.log(sig_prime_v / Pa), 1.1)

    CSR_75 = CSR / (MSF * K_sig)

    return CSR_75

def bi_find_CSR(d, M, pga, sig_v, sig_prime_v):
    ''' function bi_find_CSR to calculate CSR values from Boulanger and Idriss (2014) 
    Inputs are d (depth), M (magnitude), pga (peak ground acceleration), sig_v (vertical effective stress), sig_prime_v (effective vertical stress)
    Output is CSR (cyclic stress ratio)
    '''
    alpha = -1.012 - 1.126 * np.sin((d / 11.73) + 5.133)
    beta = 0.106 + 0.118 * np.sin((d / 11.28) + 5.142)
    r_d = np.exp(alpha + beta * M)
    CSR = 0.65 * (sig_v / sig_prime_v) * pga * r_d

    return CSR

def bi_find_CRR(M, sig_prime_v, q_c1Ncs):

    MSF = 6.9 * np.exp(- M / 4) - 0.058
    Pa = 0.101325  # MPa
    C_sig = np.minimum(1 / (37.3 - 8.27 * q_c1Ncs ** 0.264),0.3)
    K_sig = np.minimum(1 - C_sig * np.log(sig_prime_v / Pa), 1.1)
    CRR_75 = np.exp(q_c1Ncs / 113 + (q_c1Ncs / 1000) ** 2 - (q_c1Ncs / 140) ** 3 + (q_c1Ncs / 137) ** 4 - 2.8)
    CRR = CRR_75 * MSF * K_sig

    return CRR

def bi_iterate_q(q_c, sig_prime_v, sig_v, f_s, num, tol, C_FC=0):
    Pa = 0.101325  # MPa
    m0 = 1.338 - 0.249 * q_c ** 0.264  # initialize m for calculation of C_N
    C_N = np.minimum((Pa / sig_prime_v) ** m0, 1.7)
    q_c1N = C_N * q_c / Pa
    I_c = solve_Ic(q_c, sig_v, sig_prime_v, f_s)
    FC = 80 * (I_c + C_FC) - 137
    dq_c1N = (11.9 + q_c1N / 14.6) * np.exp(1.63 - 9.7 / (FC + 2) - (15.7 / (FC + 2)) ** 2)
    q_c1Ncs = q_c1N + dq_c1N

    count = 0
    while any(abs(m0 - (1.338 - 0.249 * q_c1Ncs ** 0.264)) > tol) and count <= num:
        m0 = 1.338 - 0.249 * q_c1Ncs ** 0.264
        C_N = np.minimum((Pa / sig_prime_v) ** m0,1.7)
        q_c1N = C_N * q_c / Pa
        dq_c1N = (11.9 + q_c1N / 14.6) * np.exp(1.63 - 9.7 / (FC + 2) - (15.7 / (FC + 2)) ** 2)
        q_c1Ncs = q_c1N + dq_c1N
        count += 1

    return q_c1Ncs

def bi_lpi(cpt, mags, pgas, C_FC=0, sim=None):
    ''' function bi_lpi to apply the FS/LPI method to CSR and CRR values estimated from calculations from Boulanger and Idriss (2014)
    Input cpt dictionary with CPT data
    Input mags array of magnitudes
    Input pgas dictionary with PGA values
    Input C_FC central value for fines content constant
    Input sim simulation number
    Output LPIs dictionary with LPI values '''
    num = 50
    tol = 10 ** -7
    LPIs = {}
    for key in cpt:
        d = np.array(cpt[key]['CPT_data']['d'])
        q_c = np.array(cpt[key]['CPT_data']['q_c'])
        f_s = np.array(cpt[key]['CPT_data']['f_s'])
        sig_v = np.array(cpt[key]['CPT_data']['sig_v'])
        sig_prime_v = np.array(list(map(lambda x: x[0], cpt[key]['sig_prime_v'])))
        I_c = solve_Ic(q_c, sig_v, sig_prime_v, f_s)
        nonliq = np.where(I_c > 2.6)

        if sim is not None:
            M = mags
            pga = pgas[key][sim]
            q_c1Ncs = bi_iterate_q(q_c, sig_prime_v, sig_v, f_s, num, tol, C_FC)
            CSR = bi_find_CSR(d, M, pga, sig_v, sig_prime_v)
            CRR = bi_find_CRR(M, sig_prime_v, q_c1Ncs)
            FS = np.clip(CRR / CSR, 0, 1)
            FS[nonliq] = 1
            w = 10 - 0.5 * d
            lpi = sum((1 - FS) * w * cpt[key]['CPT_data']['dz'])

        else:
            lpi = np.zeros(len(mags))
            q_c1Ncs = bi_iterate_q(q_c, sig_prime_v, sig_v, f_s, num, tol)
            w = 10 - 0.5*d
            for scen, (M,pga) in enumerate(zip(mags,pgas[key])):
                CSR = bi_find_CSR(d, M, pga, sig_v, sig_prime_v)
                CRR = bi_find_CRR(M, sig_prime_v, q_c1Ncs)
                FS = np.clip(CRR/CSR,0,1)
                FS[nonliq] = 1
                lpi[scen] = sum((1-FS)*w*cpt[key]['CPT_data']['dz'])

        LPIs[key] = lpi
    return LPIs

def solve_Ic(q_c, sig_v, sig_prime_v, f_s):
    '''solves for the soil behavior index, Ic using an interative n procedure as in Robertson (2009)
    input q_c tip resistance
    input sig_v vertical soil stress
    input sig_prime_v effective vertical soil stress
    output Ic soil behavior type index '''

    Pa = 0.101325
    n = np.full_like(sig_prime_v,1.0)  # initialize n array with same shape as sig_prime_v
    tol = 1e-3
    Q = ((q_c - sig_v) / Pa) * (Pa / sig_prime_v) ** n
    F = 100 * f_s / (q_c - sig_v)
    Q = np.maximum(1, Q)
    F = np.where(F < 0, np.maximum(1, F), F)
    Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
    n2 = 0.381*Ic + 0.05 * (sig_prime_v / Pa) - 0.15
    n2 = np.maximum(n2, 0.5)
    n2 = np.minimum(n2, 1.0)
    while any(abs(n-n2) > tol):
        Q = ((q_c - sig_v) / Pa) * (Pa / sig_prime_v) ** n2
        Q = np.maximum(1, Q)
        Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
        n = n2
        n2 = 0.381 * Ic + 0.05 * (sig_prime_v / Pa) - 0.15
        n2 = np.maximum(n2, 0.5)
        n2 = np.minimum(n2, 1.0)

    return Ic

def soil_stress(cpt, slr):
    '''return the sigma and sigma v based on cpt data
    input: cpt dict with data for each borehole
    input: slr (can be an array)
    output: cpt dict with additional data for soil stress
    '''
    # Constants
    g = 9.81
    rho_w = 1  # density of water in Mg/m^3
    gamma_w = rho_w * g / 1000  # in units of MPa
    table = {}
    # if len(cpt) > 3:
        # cpt = interp_gwt(cpt)
    for key,data in cpt.items():
        table[key] = {}
        table[key]['emergent_flag'] = np.zeros(len(slr))
        # R_f = np.zeros(len(data['CPT_data']))
        u = np.zeros(shape=(len(data['CPT_data']), len(slr)))

        for rise in range(len(slr)):
            if slr[rise] > data['Water depth']:
                table[key]['emergent_flag'][rise] = 1
                h = data['CPT_data']['d']
            else:
                h = np.maximum(data['CPT_data']['d'] - data['Water depth'] + slr[rise], 0)
            for i in range(len(u)):
                u[:, rise] = gamma_w * h

        sig_v = np.zeros(len(data['CPT_data']))
        for i in range(len(data['CPT_data'])):
            d = data['CPT_data']['d'][i]
            sig_v[i] = sum(data['CPT_data']['dsig_v'][data['CPT_data']['d'] <= d])
        sig_prime_v = u
        for i in range(u.shape[1]):
            sig_prime_v[:,i] = sig_v - u[:,i]

        data['CPT_data']['sig_v'] = sig_v
        data['sig_prime_v'] = sig_prime_v
        table[key]['wd'] = np.maximum(np.round(data['Water depth'] - np.array(slr),1), 0).tolist()

    return cpt, table

def moss_solve_FS(cpt, mags, pgas, slr, mc=None):
    '''function to determine the factor of safety against liquefaction over the depth of each borehole
    input: cpt dict with each borehole and soil data over the depth
    input: mags vector of magnitudes for each scenario
    input: pgas matrix of pgas for each simulation of each scenario
    input: slr vector of sea level rise values
    input: mc is the monte carlo simulation
    output: cpt updated dict with additional FS and CRR/CSR values
    output: depth dict with dataframe of d and dz for each borehole
    '''
    # Constants
    Pa = 0.101325
    g = 9.81

    FS = {}
    depth = {}

    if mc is not None:
        for rise in range(len(slr)):
            SLR = str(slr[rise])
            FS[SLR] = {}
            for bh in range(len(cpt)):
                key = list(cpt.keys())[bh]
                FS[SLR][key] = np.zeros(len(cpt[key]['CPT_data']))

        for bh in range(len(cpt)):
            key = list(cpt.keys())[bh]
            M = mags
            pga = pgas[key][mc]

            r_d = np.zeros(len(cpt[key]['CPT_data']))
            num = -9.147 - 4.173 * pga + 0.652 * M
            for z in range(len(cpt[key]['CPT_data'])):
                d = cpt[key]['CPT_data']['d'][z]
                den = 10.567 + 0.089 * np.exp(0.089 * (-(d * 3.28) - (7.76 * pga) + 78.576))
                rd_num = 1 + (num / den)
                den = 10.567 + 0.089 * np.exp(0.089 * (-(7.76 * pga) + 78.576))
                rd_den = 1 + (num / den)
                if d <= 20:
                    r_d[z] = rd_num / rd_den
                elif d > 20:
                    # this should never be the case
                    r_d[z] = rd_num / rd_den - 0.0014 * (d * 3.28 - 65)

            for rise in range(len(slr)):
                SLR = str(slr[rise])

                q_c1, c, C_q = iterate_c(cpt[key]['CPT_data']['q_c'], cpt[key]['CPT_data']['R_f'],
                                         cpt[key]['sig_prime_v'][:, rise])

                # CSR = np.zeros(len(cpt[key]['CPT_data']))
                CRR = np.full((len(cpt[key]['CPT_data'])), np.nan)

                CSR = 0.65 * pga * (cpt[key]['CPT_data']['sig_v'] / cpt[key]['sig_prime_v'][:, rise]) * r_d
                CSR = list(map(lambda x: max(1e-5, x), CSR)) # eliminate negative CSR values
                I_c = solve_Ic(cpt[key]['CPT_data']['q_c'], cpt[key]['CPT_data']['sig_v'],
                               cpt[key]['sig_prime_v'][:, rise], cpt[key]['CPT_data']['f_s'])

                for i in range(len(cpt[key]['CPT_data'])):
                    if I_c[i] > 2.6:
                        CRR[i] = 2 * CSR[i]  # no liquefaction
                    else:
                        try:
                            CRR[i] = np.exp(
                                (q_c1[i] ** 1.045 + q_c1[i] * (0.110 * cpt[key]['CPT_data']['R_f'][i]) +
                                 (0.001 * cpt[key]['CPT_data']['R_f'][i]) +
                                 c[i] * (1 + 0.85 * cpt[key]['CPT_data']['R_f'][i]) -
                                 0.848 * np.log(M) - 0.002 * np.log(cpt[key]['sig_prime_v'][i, rise]) - 20.923 +
                                 1.632 * norm.ppf(0.15)) / 7.177)
                        except:
                            CRR[i] = CSR[i] * 2
                FS[SLR][key] = CRR / CSR
            d = np.where(cpt[key]['CPT_data'].keys() == 'd')[0][0]
            dz = np.where(cpt[key]['CPT_data'].keys() == 'dz')[0][0]
            depth[key] = cpt[key]['CPT_data'].iloc[:, [d, dz]]

    else:

        for rise in range(len(slr)):
            SLR = str(slr[rise])
            FS[SLR] = {}
            for bh in range(len(cpt)):
                key = list(cpt.keys())[bh]
                FS[SLR][key] = np.zeros(
                    shape=(np.shape(pgas[key])[0], len(cpt[key]['CPT_data']), np.shape(pgas[key])[1]))

        for bh in range(len(cpt)):
            key = list(cpt.keys())[bh]

            for scen in range(len(pgas[key])):

                M = mags[scen]
                DWF_M = 17.84 * (M ** (-1.43))
                pga = pgas[key][scen]  # length is nsim

                r_d = np.zeros(shape=(len(cpt[key]['CPT_data']), len(pga)))
                num = -9.147 - 4.173 * pga + 0.652 * M
                for z in range(len(cpt[key]['CPT_data'])):
                    d = cpt[key]['CPT_data']['d'][z]
                    den = 10.567 + 0.089 * np.exp(0.089 * (-(d * 3.28) - (7.76 * pga) + 78.576))
                    rd_num = 1 + (num / den)
                    den = 10.567 + 0.089 * np.exp(0.089 * (-(7.76 * pga) + 78.576))
                    rd_den = 1 + (num / den)
                    if d <= 20:
                        r_d[z] = rd_num / rd_den
                    elif d > 20:
                        # this should never be the case
                        r_d[z] = rd_num / rd_den - 0.0014 * (d * 3.28 - 65)

                for rise in range(len(slr)):
                    SLR = str(slr[rise])

                    q_c1, c, C_q = iterate_c(cpt[key]['CPT_data']['q_c'], cpt[key]['CPT_data']['R_f'],
                                             cpt[key]['sig_prime_v'][:, rise])

                    CSR = np.zeros(shape=(len(cpt[key]['CPT_data']), len(pga)))
                    CRR = np.full((len(cpt[key]['CPT_data']), len(pga)), np.nan)

                    for sim in range(len(pga)):
                        CSR[:, sim] = 0.65 * (pga[sim]) * (
                                cpt[key]['CPT_data']['sig_v'] / cpt[key]['sig_prime_v'][:, rise]) * r_d[:, sim]
                        I_c = solve_Ic(cpt[key]['CPT_data']['q_c'], cpt[key]['CPT_data']['sig_v'],
                                       cpt[key]['sig_prime_v'][:, rise], cpt[key]['CPT_data']['f_s'])

                        for i in range(len(cpt[key]['CPT_data'])):
                            if (I_c[i] > 2.6):
                                CRR[i, sim] = 2 * CSR[i, sim]  # to try to match Cliq
                            else:
                                try:
                                    CRR[i, sim] = np.exp(
                                        (q_c1[i] ** 1.045 + q_c1[i] * (0.110 * cpt[key]['CPT_data']['R_f'][i]) +
                                         (0.001 * cpt[key]['CPT_data']['R_f'][i]) +
                                         c[i] * (1 + 0.85 * cpt[key]['CPT_data']['R_f'][i]) -
                                         0.848 * np.log(M) - 0.002 * np.log(cpt[key]['sig_prime_v'][i, rise]) - 20.923 +
                                         1.632 * norm.ppf(0.15)) / 7.177)
                                except:
                                    CRR[i, sim] = CSR[i, sim] * 2
                    FS[SLR][key][scen, :, :] = CRR / CSR

            d = np.where(cpt[key]['CPT_data'].keys() == 'd')[0][0]
            dz = np.where(cpt[key]['CPT_data'].keys() == 'dz')[0][0]
            depth[key] = cpt[key]['CPT_data'].iloc[:, [d, dz]]

    return depth, FS

def moss_solve_LPI(depth, FS, table):
    '''function to determine the liquefaction potential index given factor of safety values
    input: depth dict with each borehole depth and dz values
    input: FS dict of factors of safety against liquefaction for each slr scenario, borehole, scenario and simulation
    input: table dict with water depth for each slr scenario and borehole
    output: LPIs dict with liquefaction potential index for each slr scenario, borehole, scenario and simulation
    '''
    names = list(FS[list(FS.keys())[0]].keys())
    LPIs = {}
    for rise in range(len(FS)):
        SLR = list(FS.keys())[rise]
        LPIs[SLR] = {}

        for bh in range(len(names)):
            key = names[bh]
            if len(FS[SLR][key].shape) == 1:
                ndepth = len(FS[SLR][key])
                F = np.zeros(ndepth)
                for k in range(ndepth):
                    if depth[key]['d'][k] < table[key]['wd'][rise]:
                        F[k] = 0
                    elif FS[SLR][key][k] < 0:  ## added in case FS is negative then F will not be >> 1
                        F[k] = 1
                    elif FS[SLR][key][k] < 1:
                        F[k] = 1 - FS[SLR][key][k]
                tempLPI = F * depth[key]['dz'] * (10 - 0.5 * depth[key]['d'])
                LPIs[SLR][key] = sum(tempLPI)

            else:
                LPIs[SLR][key] = np.zeros(shape=(np.shape(FS[SLR][key])[0], np.shape(FS[SLR][key])[2]))

                nscen = np.shape(FS[SLR][key])[0]
                ndepth = np.shape(FS[SLR][key])[1]
                nsim = np.shape(FS[SLR][key])[2]

                for scen in range(nscen):
                    for sim in range(nsim):
                        F = np.zeros(ndepth)
                        for k in range(ndepth):
                            if depth[key]['d'][k] < table[key]['wd'][rise]:
                                F[k] = 0
                            elif FS[SLR][key][scen, k, sim] < 0:  ## added in case FS is negative then F will not be >> 1
                                F[k] = 1
                            elif FS[SLR][key][scen, k, sim] < 1:
                                F[k] = 1 - FS[SLR][key][scen, k, sim]
                        tempLPI = F * depth[key]['dz'] * (10 - 0.5 * depth[key]['d'])
                        LPIs[SLR][key][scen, sim] = sum(tempLPI)
                for i in LPIs:
                    for j in LPIs[i]:
                        rep = []
                        for k in range(len(LPIs[i][j])):
                            rep.append(LPIs[i][j][k][0])
                        LPIs[i][j] = rep

    return LPIs

def iterate_c(q_c, R_f, sig_pv):
    ''' function iterate_c to calculate c value
    input q_c tip resistance
    input R_f friction ratio
    input sig_pv effective vertical stress
    output q_c1, c_new, C_q
    '''

    # constants
    x1 = 0.78
    x2 = -0.33
    y1 = -0.32
    y2 = -0.35
    y3 = 0.49
    z1 = 1.21
    Pa = 0.101325
    tol = 1e-6

    c = np.zeros(len(R_f))
    for k in range(len(c)):
        f1 = x1 * q_c[k] ** x2
        f2 = -(y1 * (q_c[k] ** y2) + y3)
        f3 = abs(np.log10(10 + q_c[k])) ** z1
        c[k] = f1 * (R_f[k] / f3) ** f2

    C_q = (Pa / sig_pv) ** c
    C_q = np.array(list(map(lambda x: min(x, 1.7), C_q)))
    q_c1 = C_q * q_c

    c_new = np.zeros(len(c))
    for k in range(len(c_new)):
        f1 = x1 * q_c1[k] ** x2
        f2 = -(y1 * (q_c1[k] ** y2) + y3)
        f3 = abs(np.log10(10 + q_c1[k])) ** z1
        c_new[k] = f1 * (R_f[k] / f3) ** f2

    while any(abs(c - c_new) > tol):
        C_q = (Pa / sig_pv) ** c_new
        C_q = np.array(list(map(lambda x: min(x, 1.7), C_q)))
        q_c1 = C_q * q_c
        c = c_new
        for k in range(len(c_new)):
            f1 = x1 * q_c1[k] ** x2
            f2 = -(y1 * (q_c1[k] ** y2) + y3)
            f3 = abs(np.log10(10 + q_c1[k])) ** z1
            c_new[k] = f1 * (R_f[k] / f3) ** f2

    return q_c1, c_new, C_q

def soil_stress_new(cpt, slr,Kh):
    '''return the sigma and sigma v based on cpt data
    input: cpt dict with data for each borehole
    input: slr (can be an array)
    output: cpt dict with additional data for soil stress
    '''
    # Constants
    g = 9.81
    rho_w = 1  # density of water in Mg/m^3
    gamma_w = rho_w * g / 1000  # in units of MPa
    table = {}
    # if len(cpt) > 3:
        # cpt = interp_gwt(cpt)
    for key,data in cpt.items():
        table[key] = {}
        table[key]['emergent_flag'] = np.zeros(len(slr))
        # R_f = np.zeros(len(data['CPT_data']))
        u = np.zeros(shape=(len(data['CPT_data']), len(slr)))
        for rise in range(len(slr)):
            dep = data['wd']['gwt'].iloc[np.where((data['wd']['slr'] == rise) & (data['wd']['Kh'] == Kh) & (data['wd']['tide'] == 'mhhw'))[0]]
            if slr[rise] > dep[0]:
                table[key]['emergent_flag'][rise] = 1
                h = data['CPT_data']['d']
            else:
                h = np.maximum(data['CPT_data']['d'] - dep[0] + slr[rise], 0)
            u[:, rise] = gamma_w * h

        sig_v = np.zeros(len(data['CPT_data']))
        for i in range(len(data['CPT_data'])):
            d = data['CPT_data']['d'][i]
            sig_v[i] = sum(data['CPT_data']['dsig_v'][data['CPT_data']['d'] <= d])
        sig_prime_v = u
        for i in range(u.shape[1]):
            sig_prime_v[:,i] = sig_v - u[:,i]

        data['CPT_data']['sig_v'] = sig_v
        data['sig_prime_v'] = sig_prime_v
        table[key]['wd'] = np.maximum(np.round(data['wd']['slr'] - np.array(slr),1), 0).tolist()

    return cpt, table

def liqcalc_cpts(lpi,sim, fun, store, slr, mags, pgas, C_FC, SLR,Kh):
    ''' function liqcalc to run liquefaction calculations for a given simulation
    Inputs are sim (simulation number), fun (liquefaction function), store (simulation soil/cpt data), slr (sea level rise [float]), mags (magnitudes), pgas (PGA values), C_FC (fines content constant), SLR (sea level rise [string])
    Output is lpi (liquefaction potential index)
    '''

    if fun[sim] == 0:  # B&I
        soil, table = soil_stress_new(store, slr,Kh)
        lpi[sim] = bi_lpi(soil, mags[sim], pgas, C_FC[sim], sim)

    elif fun[sim] == 1:  # Moss
        soil, table = soil_stress_new(store, slr,Kh)
        depth, FS = moss_solve_FS(soil, mags[sim], pgas, slr, sim)
        lpi[sim] = moss_solve_LPI(depth, FS, table)[SLR[0]]

    else:
        return 'Problem with fun flag'

    return lpi

def make_gdf2(dflpi, cpt):
    #     names = list(cpt.keys())
    lats = np.zeros(len(dflpi))
    lons = np.zeros(len(dflpi))
    for row in range(len(dflpi)):
        # name = dflpi['bh'][row]
        # set name to the dflpi index
        name = dflpi.index[row]
        lats[row] = cpt[name]['Lat']
        lons[row] = cpt[name]['Lon']

    out = gpd.GeoDataFrame(dflpi, geometry=gpd.points_from_xy(lons, lats))

    return out

def run_lpi_sgs(points, gdflpi, sim, k = 10, rad = 5000, krig_type = 'ordinary', var_model = 'exponential'):
    '''
    function run_lpi_sgs to run the sequential Gaussian simulations for LPI across the study area
    input points: dataframe with location information
    input gdflpi: geodataframe with lpi from each simulation (at borehole points)
    input sim: simulation number to be run

    optional inputs:
    k = number of neighbors to include in SGS (default 10)
    rad = search radius in meters (default 5000 == 5km)
    krig_type = type of kriging to perform (default 'ordinary', other option: 'simple')
    var_model = the variogram model, options from scikit geostats (default 'exponential')
    
    output: points dataframe with additional LPI for each simulation
    '''
    df = pd.DataFrame(columns=['lat', 'lon', 'LPI'])
    df['lat'] = gdflpi['geometry'].y
    df['lon'] = gdflpi['geometry'].x
    df['LPI'] = gdflpi[sim]
    
    df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(df, 'lon', 'lat', 'LPI', 0.0003)
    df_grid = df_grid[df_grid["Z"].isnull() == False]
    df_grid = df_grid.rename(columns = {"Z": "LPI"})
    data = df_grid['LPI'].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
    df_grid['Nlpi'] = nst_trans.transform(data) 
    values = df_grid['Nlpi']

    coordinates,values = skg.data.pancake(N=300).get('sample')
    coords = df_grid[['X','Y']].values
    values = df_grid['Nlpi']
    V = skg.Variogram(coords, values, model=var_model)

    Pred_grid_xy = gs.Gridding.prediction_grid(min(points['lon']), max(points['lon']), min(points['lat']), max(points['lat']), 0.001)  ## the resolution effects computation time!

    lon = Pred_grid_xy[:,0]
    lat = Pred_grid_xy[:,1]
    lon = np.array(lon)
    lat = np.array(lat)
    lon = np.unique(lon)
    lat = np.unique(lat)
    rows = len(lat)
    cols = len(lon)
    # variogram parameters:
    azimuth = 0
    nugget = V.parameters[2]
    major_range = V.parameters[0]
    minor_range = V.parameters[0]
    sill = V.parameters[1]
    vtype = 'Exponential'
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    if krig_type == 'simple':
        sim1 = gs.Interpolation.skrige_sgs(Pred_grid_xy, df_grid, 'X', 'Y', 'Nlpi', k, vario, rad)
    elif krig_type == 'ordinary':
        sim1 = gs.Interpolation.okrige_sgs(Pred_grid_xy, df_grid, 'X', 'Y', 'Nlpi', k, vario, rad)
    else: 
        print('Invalid krig_type. Running ordinary kriging')
        sim1 = gs.Interpolation.okrige_sgs(Pred_grid_xy, df_grid, 'X', 'Y', 'Nlpi', k, vario, rad)
    sim1 = sim1.reshape(-1,1)
    sim_trans = nst_trans.inverse_transform(sim1)

    distances = cdist(points[['lon', 'lat']], Pred_grid_xy)
    closest_indices = np.argmin(distances, axis=1)
    points[sim] = sim_trans[closest_indices]

    return points

def geyin_ff(LPI):
    '''
    function geyin_ff calculates the loss ratio from LPI based on Geyin et al. 2020
    input: LPI as a single value or array
    output: loss ratio of the same size as LPI
    '''
    loss_ratio = 0.11668 * (LPI**0.4607)
    return loss_ratio

def liq_frag_func(LPI):
    '''
    function liq_frag_func calculates fragility function for a 2D input of LPI values
    inputs: array of LPIs
    outputs: array of loss_ratios
    ''' 
    loss_ratio = []
    for i in range(LPI.shape[0]):
        temp = list(map(lambda x:geyin_ff(x),LPI[i]))
        loss_ratio.append(temp)
    loss_ratio = np.array(loss_ratio)
    
    return loss_ratio