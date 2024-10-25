## adapt_retrofit.py to run the seismic retrofit adaptation strategy across all hazards
## By Emily Mongold 08/09/2024

import numpy as np
import pandas as pd
import geopandas as gpd
from NNR import NNR
import matplotlib.pyplot as plt
import os
import pickle
from regional_mh_risk.postprocess import get_pM_pypsha, get_pliq
from regional_mh_risk.simple_liquefaction import liq_frag_func
from regional_mh_risk.earthquake import eq_shaking_loss
import shapely.wkt
import contextily as ctx
import copy

# where the results from pypsha are stored for earthquake
eqoutdir = './ground_motions/new_gms/'

# load the outputs from each individual hazard:
# coastal flooding
with open('flood_bldg_loss_new.pkl','rb') as f:
    flood_bldg_loss = pickle.load(f)
flood_bldgs = pd.read_csv('flood_bldg_outs_new.csv')
# earthquake
earthquake_bldgs = pd.read_csv('eq_bldgs_out_nsi_hazus.csv')
# tsunami
with open('tsunami_bldg_loss.pkl','rb') as f:
    tsunami_bldg_loss = pickle.load(f)
with open('tsunami_bldg_LRs.pkl','rb') as f:
    tsunami_bldg_LRs = pickle.load(f)
tsunami_bldgs = pd.read_csv('tsunami_bldg_outs_nsi.csv')

print('loaded inputs')

# find the oldest buildings indices
print('age of 40th percentile building',earthquake_bldgs['YearBuilt'].quantile(0.4))
oldest_40_eq_bldgs = earthquake_bldgs[earthquake_bldgs['YearBuilt'] <= earthquake_bldgs['YearBuilt'].quantile(0.4)].index
earthquake_bldgs['YearBuilt_retrofit40pct'] = earthquake_bldgs['YearBuilt']
earthquake_bldgs.loc[oldest_40_eq_bldgs,'YearBuilt_retrofit40pct'] = 2020
earthquake_bldgs['geometry'] = earthquake_bldgs['geometry'].apply(shapely.wkt.loads)
earthquake_bldgs = gpd.GeoDataFrame(earthquake_bldgs, geometry='geometry', crs='EPSG:4326')

plt.figure(figsize=(8,8))
ax=earthquake_bldgs.loc[oldest_40_eq_bldgs].plot(color='purple', markersize=1)
plt.legend(['Buildings to retrofit'],markerscale=5)
plt.ylim(37.725, 37.79)
plt.xlim(-122.3, -122.22)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, crs='EPSG:4326')  #Positron  #DarkMatter  #Voyager
plt.savefig('figures/figure9c.png',format='png', dpi=1000, bbox_inches='tight')

print('saved selected buildings figure')

## savedir should contain the outputs of the liquefaction run
savedir = './all_alameda_geology/'
files = os.listdir(savedir)
slrs = [0.0,0.25,0.5,0.75,1.0,1.25,1.5]
points = {}
for slr in slrs:
    csv_files = [file for file in files if file.endswith(str(slr) + '.csv')]
    for i in range(len(csv_files)):
        if i == 0:
            points[slr] = pd.read_csv(savedir + csv_files[i])
        else:
            file_path = os.path.join(savedir, csv_files[i])
            try:
                df = pd.read_csv(file_path).drop(columns=['utmX','utmY','lat','lon','Unnamed: 0','geometry'])
            except:
                df = pd.read_csv(file_path).drop(columns=['utmX','utmY','lat','lon','geometry'])
            points[slr] = pd.concat([points[slr], df], axis=1)
for slr in points.keys():
    points[slr]['geometry'] = points[slr]['geometry'].apply(lambda x: shapely.wkt.loads(x))

output = {}
sims = {}
for slr in points.keys():
    output[slr] = points[slr].drop(columns=['index_right','lat','lon','utmX','utmY','geometry','id','AREA','PERIMETER','SFQ2_','SFQ2_ID','PTYPE','PTYPE2','LIQ','LIQ_SOURCE'],axis=1)
    # change column names to integers of their name
    output[slr].columns = [int(i) for i in output[slr].columns]
    # sort output by columns 0-2423
    try: 
        output[slr] = output[slr].reindex(sorted(output[slr].columns), axis=1)
    except:
        # print duplicated columns
        print(slr)
        print(output[slr].columns[output[slr].columns.duplicated()])
    sims[slr] = output[slr].columns
    output[slr] = np.swapaxes(np.array(output[slr]),0,1)
    nsim = output[slr].shape[0]

## obtain the return period of the ruptures and the probability of liquefaction
lambda_M = get_pM_pypsha(eqoutdir+'event_save_newvs30.pickle')
pliq = {}
for slr in output.keys():
    pliq[slr] = get_pliq(output[slr])
## replace nan with 0
for slr in pliq.keys():
    pliq[slr] = np.nan_to_num(pliq[slr])
## calculate the return period of liquefaction
lambda_liq = {}
for slr in output.keys():
    temp = []
    for i,sim in enumerate(sims[slr]):
        temp.append(pliq[slr][i,:]*lambda_M[sim])
    lambda_liq[slr] = sum(temp) 

building_LPIs = {}

y_t = list(earthquake_bldgs['Latitude'])
x_t = list(earthquake_bldgs['Longitude'])
for slr in slrs:
    print('running ',slr)
    X = np.array(points[slr]['lon'])
    Y = np.array(points[slr]['lat'])
    Z = output[slr].T
    z_t = NNR(np.column_stack((x_t, y_t)), np.column_stack((X, Y)), Z, sample_size=-1, n_neighbors=4, weight='distance2')
    bldLPI = z_t.T
    building_LPIs[slr] = bldLPI

pga = pd.read_csv(eqoutdir + 'newvs30_pgas_pypsha.csv',header=None)
locs = pd.read_csv(eqoutdir + 'Alameda_new_vs30.csv', index_col=0)
X = np.array(locs['lon'])
Y = np.array(locs['lat'])
Z = np.array(np.array(pga.T))
z_t = NNR(np.column_stack((x_t, y_t)), np.column_stack((X, Y)), Z, sample_size=-1, n_neighbors=4, weight='distance2')
building_PGAs = z_t.T

print('calculated LPIs and pgas')

retrofit_blds = copy.deepcopy(earthquake_bldgs)
retrofit_blds['YearBuilt'] = retrofit_blds['YearBuilt_retrofit40pct']
LR_gs = eq_shaking_loss(earthquake_bldgs, building_PGAs)
LR_gs_40 = eq_shaking_loss(retrofit_blds, building_PGAs)
print(LR_gs.max())
reduction = LR_gs - LR_gs_40
print('ran for ground shaking')

# plot the change in LR distributions
plt.figure()
plt.hist(LR_gs.flatten(),alpha=0.5)
plt.hist(LR_gs_40.flatten(),alpha=0.5)
plt.legend(['Base','Retrofit'])
plt.title('LR_gs distributions')
plt.show()

LR = {}
LR_40ret = {}
for slr in building_LPIs.keys():
    LR_liq = liq_frag_func(building_LPIs[slr])
    LR_liq = np.nan_to_num(LR_liq) #replace NaN with 0
    LR_liq_ret = (LR_liq - reduction).clip(min=0)  ## remove the savings from retrofitting
    LR[slr] = np.maximum(LR_gs[sims[slr],:],LR_liq)
    LR_40ret[slr] = np.maximum(LR_gs_40[sims[slr],:],LR_liq_ret) # try with the reduction 

print('calculated LR')

eq_bldg_losses_base = {}
eq_reg_losses_base = {}
eq_bldg_losses_ret = {}
eq_reg_losses_ret = {}
for slr in LR.keys():
    LR_both_base = LR[slr].clip(max=1)
    LR_both_ret = LR_40ret[slr].clip(max=1)
    eq_bldg_losses_base[slr] = []
    eq_bldg_losses_ret[slr] = []
    for sim in range(LR_both_base.shape[0]):
        temp = LR_both_base[sim,:] * earthquake_bldgs['ImprovementValue']
        eq_bldg_losses_base[slr].append(temp)
        temp = LR_both_ret[sim,:] * earthquake_bldgs['ImprovementValue']
        eq_bldg_losses_ret[slr].append(temp)
    eq_bldg_losses_base[slr] = np.array(eq_bldg_losses_base[slr])
    eq_reg_losses_base[slr] = sum(eq_bldg_losses_base[slr].T)
    eq_bldg_losses_ret[slr] = np.array(eq_bldg_losses_ret[slr])
    eq_reg_losses_ret[slr] = sum(eq_bldg_losses_ret[slr].T)

losses = np.arange(0, 5e9, 5e5) # total portfolio is 7e9

lambda_M = get_pM_pypsha(eqoutdir+'event_save_newvs30.pickle')
exceedance_earthquake = pd.DataFrame()
exceedance_earthquake_ret = pd.DataFrame()
for slr in eq_bldg_losses_base.keys():
    temp = []
    temp_ret = []
    regloss = np.sum(eq_bldg_losses_base[slr], axis=1)
    regloss_ret = np.sum(eq_bldg_losses_ret[slr], axis=1)
    for loss in losses:
        where = np.where(regloss > loss)
        temp.append(np.sum(lambda_M[where]))
        where = np.where(regloss_ret > loss)
        temp_ret.append(np.sum(lambda_M[where]))
    exceedance_earthquake[slr] = temp
    exceedance_earthquake_ret[slr] = temp_ret

## for tsunami and coastal flooding (unaffected)
num_RPs = [72,100,200,475,975,2475,3000]
nsamples = len(tsunami_bldg_loss[next(iter(tsunami_bldg_loss))].iloc[0,0])
loss_tsunami_tot = pd.concat(tsunami_bldg_loss.values(), axis=0).groupby(level=0).sum()
loss_tsunami_tot.columns = 1/np.array(loss_tsunami_tot.columns.astype(int))
exceedance_tsunami = pd.DataFrame()
lambdas = 1/np.array(num_RPs)
occurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
occurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
occurrence[-1] = lambdas[-1]
for slr in loss_tsunami_tot.index:
    temp = []
    for loss in losses:
        rate = loss_tsunami_tot.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_tsunami[slr] = temp

loss_flood_tot = pd.concat(flood_bldg_loss.values(), axis=0).groupby(level=0).sum()
loss_flood_tot.columns = 1/np.array(loss_flood_tot.columns)

nsamples = len(flood_bldg_loss[next(iter(flood_bldg_loss))].iloc[0,0])
exceedance_coastal_flood = pd.DataFrame()
lambdas = 1/np.array(flood_bldg_loss[next(iter(flood_bldg_loss))].columns)
occurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
occurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
occurrence[-1] = lambdas[-1]
for slr in flood_bldg_loss[next(iter(flood_bldg_loss))].index:
    temp = []
    for loss in losses:
        rate = loss_flood_tot.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_coastal_flood[slr] = temp

## combine for total loss exceedance
exceedance_total = pd.DataFrame(columns=exceedance_earthquake.columns, index=losses)
exceedance_total_ret = pd.DataFrame(columns=exceedance_earthquake.columns, index=losses)
for slr in exceedance_earthquake.columns:
    slr_to_inches = {0.0: 0, 0.25: 12, 0.5: 18, 0.75: 30, 1.0: 39, 1.25: 48, 1.5: 60} # 1.0m can also map to 42 inches
    slr_in = slr_to_inches.get(slr)
    exceedance_total.loc[:,slr] = (np.array(exceedance_earthquake[slr]) + np.array(exceedance_coastal_flood[slr_in]) + exceedance_tsunami[slr]).values
    exceedance_total_ret.loc[:,slr] = (np.array(exceedance_earthquake_ret[slr]) + np.array(exceedance_coastal_flood[slr_in]) + exceedance_tsunami[slr]).values

print('calculated exceedance')
print('difference',exceedance_total[0.0] - exceedance_total_ret[0.0])

expected_annual_loss_ret = {}
expected_annual_loss = {}
for slr_scenario in exceedance_total_ret.columns:
    exceedance_rates = exceedance_total_ret[slr_scenario].values
    expected_annual_loss_ret[slr_scenario] = np.trapz(exceedance_rates, losses)
    exceedance_rates = exceedance_total[slr_scenario].values
    expected_annual_loss[slr_scenario] = np.trapz(exceedance_rates, losses)

print('calculated EAL')

## save the values for mitigation comparison
with open('adaptation_retrofit_both.pkl','wb') as f:
    pickle.dump({'baseline_eal':expected_annual_loss,'retrofit40pct': expected_annual_loss_ret,
                 'exceedance_base':exceedance_total,'exceedance_retrofit':exceedance_total_ret},f)

print('saved and done')