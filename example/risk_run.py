## risk.py to run risk calculations compiling all hazards
## by Emily Mongold
## 08/08/2024

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import contextily as ctx
import scipy.stats as stats
from scipy.optimize import curve_fit
import pickle
from regional_mh_risk.postprocess import get_pM_pypsha
import shapely.wkt

# where the results from pypsha are stored for earthquake
eqoutdir = './ground_motions/new_gms/'

# execute fig_pre.py
exec(open('./fig_pre.py').read())

print('finished imports')

# load the outputs from each individual hazard:
# coastal flooding
with open('flood_bldg_loss_new.pkl','rb') as f:
    flood_bldg_loss = pickle.load(f)
flood_bldgs = pd.read_csv('flood_bldg_outs_new.csv')

# earthquake, this time using whitman LR from DS
with open('eq_bldg_loss_nsi_hazus.pkl','rb') as f:
    earthquake_bldg_loss = pickle.load(f)
earthquake_bldgs = pd.read_csv('eq_bldgs_out_nsi_hazus.csv')

# tsunami
with open('tsunami_bldg_loss.pkl','rb') as f:
    tsunami_bldg_loss = pickle.load(f)
with open('tsunami_bldg_LRs.pkl','rb') as f:
    tsunami_bldg_LRs = pickle.load(f)
tsunami_bldgs = pd.read_csv('tsunami_bldg_outs_nsi.csv')

tsunami_bldgs.drop(columns=['Unnamed: 0', 'OccupancyClass'], inplace=True)
earthquake_bldgs.drop(columns=['Unnamed: 0', 'Units', 'ImprovementValue', 'YearBuilt', 'Stories', 'Latitude', 'Longitude',
       'OccupancyClass', 'geometry'], inplace=True)
flood_bldgs.drop(columns=['Unnamed: 0', 'Units', 'ImprovementValue', 'YearBuilt', 'Stories', 'Latitude', 'Longitude',
       'OccupancyClass', 'geometry', 'utmX', 'utmY'], inplace=True)
all_bldgs = pd.DataFrame()
all_bldgs['geometry'] = tsunami_bldgs['geometry'].apply(shapely.wkt.loads)
all_bldgs_gdf = gpd.GeoDataFrame(all_bldgs, geometry=all_bldgs['geometry'],crs='EPSG:4326')
tsunami_bldgs.loc[:,tsunami_bldgs.columns[tsunami_bldgs.columns.str.startswith('EAL_')]] = tsunami_bldgs.loc[:,tsunami_bldgs.columns[tsunami_bldgs.columns.str.startswith('EAL_')]].fillna(0)

print('loaded the data')
BRV = tsunami_bldgs['ImprovementValue']
portfolio_value = tsunami_bldgs['ImprovementValue'].sum()
print('portfolio value: ',portfolio_value)

## for coastal flood
losses = np.arange(0, 5e9, 5e5) # total portfolio is 7e9
loss_flood_tot = pd.concat(flood_bldg_loss.values(), axis=0).groupby(level=0).sum()
loss_flood_tot.columns = 1/np.array(loss_flood_tot.columns)
nsamples = len(flood_bldg_loss[next(iter(flood_bldg_loss))].iloc[0,0])
exceedance_coastal_flood = pd.DataFrame()
lambdas = 1/np.array(flood_bldg_loss[next(iter(flood_bldg_loss))].columns)
# make lambdas the difference between the lambda values so they sum to 1
occurrence = np.zeros_like(lambdas)  # Initialize array of same shape as lambdas
occurrence[:-1] = lambdas[:-1] - lambdas[1:]  # Calculate differences
occurrence[-1] = lambdas[-1]
for slr in flood_bldg_loss[next(iter(flood_bldg_loss))].index:
    temp = []
    for loss in losses:
        rate = loss_flood_tot.loc[slr].apply(lambda x:sum(np.array(x) > loss)) * (occurrence) * (1/nsamples)
        temp.append(sum(rate))
    exceedance_coastal_flood[slr] = temp

## for earthquake
lambda_M = get_pM_pypsha(eqoutdir+'event_save_newvs30.pickle')
exceedance_earthquake = pd.DataFrame()
for slr in earthquake_bldg_loss.keys():
    temp = []
    regloss = np.sum(earthquake_bldg_loss[slr], axis=1)
    for loss in losses:
        where = np.where(regloss > loss)
        temp.append(np.sum(lambda_M[where]))
    exceedance_earthquake[slr] = temp

## for tsunami
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

## combine for total loss exceedance
exceedance_total = pd.DataFrame(columns=exceedance_earthquake.columns, index=losses)
for slr in exceedance_earthquake.columns:
    slr_to_inches = {0.0: 0, 0.25: 12, 0.5: 18, 0.75: 30, 1.0: 39, 1.25: 48, 1.5: 60} # 1.0m is the mean of 36 and 42 inches
    slr_in = slr_to_inches.get(slr)
    exceedance_total.loc[:,slr] = (np.array(exceedance_earthquake[slr]) + np.array(exceedance_coastal_flood[slr_in]) + exceedance_tsunami[slr]).values

print('calculated exceedance')
print('generating loss exceedance plots')

plt.figure(figsize=(10, 6))

# Plot for 'exceedance_total' with grey colors
cmap_blk = mpl.cm.get_cmap('Greys')
colors_blk = cmap_blk(np.linspace(0.3, 1, len(exceedance_total.columns.astype(int))))
for i, column in enumerate(exceedance_total.columns.astype(float)):
    plt.semilogy(losses/1e9, exceedance_total[column], color=colors_blk[i], label=column)
plt.xlabel('Regional loss [Billion USD]')
plt.ylabel('Exceedance rate')
plt.ylim(1e-4, 2e0)
plt.legend(title='SLR [m]').get_title().set_fontsize('14')
plt.savefig('figures/figure4a.png', format='png', dpi=1000, bbox_inches='tight')

## plot together and separated by hazard
plt.figure(figsize=(10, 6))
cmap_blk = mpl.cm.get_cmap('Greys')
colors_blk = cmap_blk(np.linspace(0.3, 1, len(exceedance_total.columns)))
for i, column in enumerate(exceedance_total.columns):
    plt.semilogy(losses/1e9, exceedance_total[column], color=colors_blk[i], label=column)
# add legend with darkest color for each hazard
plt.plot([], [], linewidth = 3,color='white', label='Hazard type')
plt.plot([], [], linewidth = 3, color='black', label='All hazards')
plt.plot([], [], linewidth = 3, color='steelblue', label='Coastal flood')
plt.plot([], [], linewidth = 3, color='red', label='Earthquake')
plt.plot([], [], linewidth = 3, color='orange', label='Tsunami')
plt.legend(title='SLR [m]').get_title().set_fontsize('12')
# Plot for 'exceedance_flood' with blue colors
cmap_blue = mpl.cm.get_cmap('Blues')
colors_blue = cmap_blue(np.linspace(0.3, 1, len(exceedance_coastal_flood.columns)))
for i, column in enumerate(exceedance_coastal_flood.columns):
    plt.semilogy(losses/1e9, exceedance_coastal_flood[column], color=colors_blue[i], label=column)
# Plot for 'exceedance_earthquake' with red colors
cmap_red = mpl.cm.get_cmap('Reds')
colors_red = cmap_red(np.linspace(0.3, 1, len(exceedance_earthquake.columns)))
for i, column in enumerate(exceedance_earthquake.columns):
    plt.semilogy(losses/1e9, exceedance_earthquake[column], color=colors_red[i], label=column)
# Plot for 'exceedance_tsunami' with orange colors
cmap_orange = mpl.cm.get_cmap('Oranges')
colors_orange = cmap_orange(np.linspace(0.3, 1, len(exceedance_tsunami.columns)))
for i, column in enumerate(exceedance_tsunami.columns):
    plt.semilogy(losses/1e9, exceedance_tsunami[column], color=colors_orange[i], label=column)

plt.xlabel('Regional loss [Billion USD]')
plt.ylabel('Exceedance rate')
plt.ylim(1e-4, 2e0)
plt.savefig('figures/figure5a.png', format='png', dpi=1000, bbox_inches='tight')

print('calculating aal')

## Building level aal
## Convert the shape of flood_bldg_loss
slrs = list(earthquake_bldg_loss.keys())
cf_bldgs = list(flood_bldg_loss.keys())
return_periods = list(flood_bldg_loss[cf_bldgs[0]].keys())
cf_bldg_loss = {slr: np.zeros((len(return_periods),len(cf_bldgs))) for slr in slrs}

for bld_index, bld in enumerate(cf_bldgs):
    for rp_index, rp in enumerate(return_periods):
        for ind,loss in enumerate(flood_bldg_loss[bld][rp]):
            cf_bldg_loss[slrs[ind]][rp_index, bld_index] = loss.mean()

## convert the shape of tsunami_bldg_loss
slrs = list(earthquake_bldg_loss.keys())
tsu_bldgs = list(tsunami_bldg_loss.keys())
return_periods = list(tsunami_bldg_loss[tsu_bldgs[0]].keys())
tsu_bldg_loss = {slr: np.zeros((len(return_periods),len(tsu_bldgs))) for slr in slrs}

for bld_index, bld in enumerate(tsu_bldgs):
    for rp_index, rp in enumerate(return_periods):
        for ind,loss in enumerate(tsunami_bldg_loss[bld][rp]):
            tsu_bldg_loss[slrs[ind]][rp_index, bld_index] = loss.mean()

## Building level aal
eq_eal = {}
cf_eal = {}
tsu_eal = {}
lambda_cf = 1/np.array(flood_bldg_loss[next(iter(flood_bldg_loss))].columns)
lambda_tsu = 1/np.array(num_RPs)
for slr in earthquake_bldg_loss.keys():
    eq_eal[slr] = np.sum(earthquake_bldg_loss[slr].T*lambda_M,axis=1)
    cf_eal[slr] = np.sum(cf_bldg_loss[slr].T*lambda_cf,axis=1)
    tsu_eal[slr] = np.sum(tsu_bldg_loss[slr].T*lambda_tsu,axis=1)
    eal_temp = np.array(eq_eal[slr])
    eal_temp[tsu_bldgs] += tsu_eal[slr]
    eal_temp[cf_bldgs] += cf_eal[slr]

expected_annual_loss = {}
for slr_scenario in exceedance_total.columns:
    exceedance_rates = exceedance_total[slr_scenario].values
    expected_annual_loss[slr_scenario] = np.trapz(exceedance_rates, losses)

eal_flood = {}
for slr_scenario in exceedance_coastal_flood.columns:
    exceedance_rates = exceedance_coastal_flood[slr_scenario].values
    eal_flood[slr_scenario] = np.trapz(exceedance_rates, losses)

eal_eq = {}
for slr_scenario in exceedance_earthquake.columns:
    exceedance_rates = exceedance_earthquake[slr_scenario].values
    eal_eq[slr_scenario] = np.trapz(exceedance_rates, losses)

eal_tsunami = {}
for slr_scenario in exceedance_tsunami.columns:
    exceedance_rates = exceedance_tsunami[slr_scenario].values
    eal_tsunami[slr_scenario] = np.trapz(exceedance_rates, losses)

slr_levels = [str(x) for x in expected_annual_loss.keys()]
eal = list(expected_annual_loss.values())

print('plotting aal')

# Create bar plot of total EAL for each SLR scenario
fig, ax = plt.subplots(figsize=(8, 6))  # You might need to adjust the figsize
bars = ax.bar(slr_levels, np.array(eal)/1e6, color='black')
ax.set_xlabel('Sea Level Rise (SLR) Amount [m]')
ax.set_ylabel('Average Annual Loss [Million USD]')
plt.xticks(rotation=0)  # Ensure SLR scenario names are horizontal to avoid overlap
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.savefig('figures/figure4b.png', format='png', dpi=1000, bbox_inches='tight')


# Create a stacked bar plot of normalized EAL per hazard
slr_levels = [str(x) for x in expected_annual_loss.keys()]
flood_losses_norm = [eal_flood[slr_to_inches.get(key)] / expected_annual_loss[key] for key in expected_annual_loss]
earthquake_losses_norm = [eal_eq[key] / expected_annual_loss[key] for key in expected_annual_loss]
tsunami_losses_norm = [eal_tsunami[key] / expected_annual_loss[key] for key in expected_annual_loss]
eal = list(expected_annual_loss.values())
fig, ax = plt.subplots(figsize=(8, 6))  ## use 10,6 with outside legend
bars_flood = ax.bar(slr_levels, flood_losses_norm, label='Flood', color='steelblue')
bars_tsunami = ax.bar(slr_levels, tsunami_losses_norm, bottom=flood_losses_norm, label='Tsunami', color='orange')
bars_earthquake = ax.bar(slr_levels, earthquake_losses_norm, bottom=np.array(flood_losses_norm) + np.array(tsunami_losses_norm), label='Earthquake', color='red')
ax.set_xlabel('Sea Level Rise (SLR) Amount [m]')
ax.set_ylabel('$AAL_{norm}$')
handles, labels = ax.get_legend_handles_labels()
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/figure5b.png', format='png', dpi=1000, bbox_inches='tight')

print('saved aal plots')

all_bldgs_gdf['total_eal_0m'] = (tsunami_bldgs['EAL_0'] + earthquake_bldgs['EAL_0.0'] + flood_bldgs['flood_eal_00'])
all_bldgs_gdf['total_aal_0.25m'] = (tsunami_bldgs['EAL_0.25'] + earthquake_bldgs['EAL_0.25'] + flood_bldgs['flood_eal_25'])
all_bldgs_gdf['total_eal_0.75m'] = (tsunami_bldgs['EAL_0.75'] + earthquake_bldgs['EAL_0.75'] + flood_bldgs['flood_eal_75'])
all_bldgs_gdf['delta_EAL_0.75'] = all_bldgs_gdf['total_eal_0.75m'] - all_bldgs_gdf['total_eal_0m']
all_bldgs_gdf['delta_AAL_0.25'] = all_bldgs_gdf['total_aal_0.25m'] - all_bldgs_gdf['total_eal_0m']

# Calculate some AALR and delta AALR
all_bldgs_gdf['AALR_0m'] = all_bldgs_gdf['total_eal_0m'] / tsunami_bldgs['ImprovementValue']
all_bldgs_gdf['delta_AALR_0.25m'] = all_bldgs_gdf['delta_AAL_0.25'] / tsunami_bldgs['ImprovementValue']
all_bldgs_gdf['delta_AALR_0.75m'] = all_bldgs_gdf['delta_EAL_0.75'] / tsunami_bldgs['ImprovementValue']

ax = all_bldgs_gdf.plot(column='AALR_0m', cmap=davos_cmap, legend=True, legend_kwds={'label': "AALR_{0}",'orientation':'vertical'},
               markersize=1,norm=mpl.colors.Normalize(vmin=all_bldgs_gdf['AALR_0m'].min(), vmax=all_bldgs_gdf['AALR_0m'].max()))
ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.2f'))
ax.set_xticks(ax.get_xticks()[::2])
plt.ylim(37.725, 37.79)
plt.xlim(-122.3, -122.22)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, crs='EPSG:4326')  #Positron  #DarkMatter  #Voyager
plt.savefig('figures/figure3.png', format='png', dpi=1000, bbox_inches='tight')


ax = all_bldgs_gdf[all_bldgs_gdf['delta_AALR_0.25m']>0.01].plot(column='delta_AALR_0.25m', cmap=davos_cmap, legend=True, legend_kwds={'label': "$\Delta AALR_{0.25m}$ [USD]",'orientation':'vertical'},
               markersize=1,norm=mpl.colors.Normalize(vmin=0.01, vmax=all_bldgs_gdf['delta_AALR_0.25m'].max()))
ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.2f'))
plt.ylim(37.725, 37.79)
plt.xlim(-122.3, -122.22)
ax.set_xticks(ax.get_xticks()[::2])
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, crs='EPSG:4326',)  #Positron  #DarkMatter  #Voyager
plt.savefig('figures/figure6a.png', format='png', dpi=1000, bbox_inches='tight')


ax = all_bldgs_gdf[all_bldgs_gdf['delta_AALR_0.75m']>0.01].plot(column='delta_AALR_0.75m', cmap=davos_cmap, legend=True, legend_kwds={'label': "$\Delta AALR_{0.75m}$ [USD]",'orientation':'vertical'},
               markersize=1 ,norm=mpl.colors.Normalize(vmin=0.01, vmax=all_bldgs_gdf['delta_AALR_0.25m'].max()))
ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.2f'))
ax.set_xticks(ax.get_xticks()[::2])
plt.ylim(37.725, 37.79)
plt.xlim(-122.3, -122.22)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, crs='EPSG:4326',)  #Positron  #DarkMatter  #Voyager
plt.savefig('figures/figure6b.png', format='png', dpi=1000, bbox_inches='tight')

print('saved building-level aal maps')
print('calculating SLR and AAL over time')

def func(x, a, b, c):
    return a * np.exp(b * x) + c

xdata = np.array([float(x) for x in slr_levels]) # converted to cm
ydata = np.array(eal)
popt, pcov = curve_fit(func, xdata, ydata,maxfev=1000)

## from https://sealevel.nasa.gov/task-force-scenario-tool?psmsl_id=437
## percentiles relating to SLR change in future years -- intermediate to align with literature & locally reported projections
data_low = {
    "2020": [21.190157, 36.190155, 54.190155], "2030": [38.190155, 59.190155, 83.190155],
    "2040": [56.190155, 82.190155, 113.190155], "2050": [73.190155, 98.190155, 135.19016],
    "2060": [90.190155, 119.190155, 159.19016], "2070": [100.190155, 140.19016, 189.19016],
    "2080": [111.190155, 154.19016, 208.19016], "2090": [123.190155, 169.19016, 232.19016],
    "2100": [125.190155, 189.19016, 271.19016], "2110": [128.19016, 202.19016, 298.19016],
    "2120": [133.19016, 214.19016, 323.19016], "2130": [137.19016, 228.19016, 351.19016],
    "2140": [140.19016, 240.19016, 378.19016], "2150": [143.19016, 251.19016, 407.19016],
}
data_int = {
    "2020": [32.190155, 55.190155, 78.190155], "2030": [63.190155, 86.190155, 116.190155],
    "2040": [95.190155, 131.19016, 179.19016], "2050": [142.19016, 190.19016, 261.19016],
    "2060": [195.19016, 262.19016, 360.19016], "2070": [269.19016, 351.19016, 473.19016],
    "2080": [376.19016, 474.19016, 610.1902], "2090": [520.1902, 648.1902, 758.1902],
    "2100": [639.1902, 839.1902, 957.1902], "2110": [754.1902, 1039.1902, 1240.1902],
    "2120": [865.1902, 1202.1902, 1631.1902], "2130": [965.1902, 1355.1902, 2142.1902],
    "2140": [1060.1902, 1501.1902, 2775.1902], "2150": [1145.1902, 1655.1902, 3534.1902],
}
data_high = {
    "2020": [34.190155, 57.190155, 80.190155], "2030": [65.190155, 106.190155, 157.19016],
    "2040": [117.190155, 190.19016, 297.19016], "2050": [217.19016, 331.19016, 484.19016],
    "2060": [374.19016, 540.1902, 710.1902], "2070": [591.1902, 822.1902, 984.1902],
    "2080": [905.1902, 1154.1902, 1313.1902], "2090": [1288.1902, 1518.1902, 1676.1902],
    "2100": [1665.1902, 1884.1902, 2061.1902], "2110": [1945.1902, 2273.1902, 2504.1902],
    "2120": [2141.1902, 2599.1902, 3084.1902], "2130": [2327.1902, 2868.1902, 3748.1902],
    "2140": [2460.1902, 3144.1902, 4422.19], "2150": [2561.1902, 3399.19, 5115.19],
}

# Convert to DataFrames with 'Percentile' as the index
low = pd.DataFrame(data_low, index=[17, 50, 83])
Int = pd.DataFrame(data_int, index=[17, 50, 83])
high = pd.DataFrame(data_high, index=[17, 50, 83])

# make lognormal fits for each of these data sets
low.loc[5] = 0
low.loc[95] = 0
Int.loc[5] = 0
Int.loc[95] = 0
high.loc[5] = 0
high.loc[95] = 0
percentiles = low.index/100
p_loss = pd.DataFrame(index=['Low','Int','High'],columns=low.columns)
e_p_loss = pd.DataFrame(index=['Low','Int','High'],columns=low.columns)
for col in low:
    values_low = low[col].values
    values_int = Int[col].values
    values_high = high[col].values
    sigma_low = ((np.log(values_low)[2] - np.log(values_low)[1])+(np.log(values_low)[1] - np.log(values_low)[0]))/(2*0.96)
    sigma_int = ((np.log(values_int)[2] - np.log(values_int)[1])+(np.log(values_int)[1] - np.log(values_int)[0]))/(2*0.96)
    sigma_high = ((np.log(values_high)[2] - np.log(values_high)[1])+(np.log(values_high)[1] - np.log(values_high)[0]))/(2*0.96)

    lognormal_low = stats.lognorm(s=sigma_low, loc=0, scale=values_low[1])
    lognormal_int = stats.lognorm(s=sigma_int, loc=0, scale=values_int[1])
    lognormal_high = stats.lognorm(s=sigma_high, loc=0, scale=values_high[1])

    low.loc[5,col] = lognormal_low.ppf(0.05)
    low.loc[95,col] = lognormal_low.ppf(0.95)
    Int.loc[5,col] = lognormal_int.ppf(0.05)
    Int.loc[95,col] = lognormal_int.ppf(0.95)
    high.loc[5,col] = lognormal_high.ppf(0.05)
    high.loc[95,col] = lognormal_high.ppf(0.95)
low = low.sort_index()
Int = Int.sort_index()
high = high.sort_index()
# plot the SLR scenarios over time
plt.figure()
plt.plot(low.columns.astype(int),low.loc[50].values,c='k')
plt.fill_between(low.columns.astype(int),low.loc[17],low.loc[83],color='k',alpha=0.2)
plt.fill_between(low.columns.astype(int),low.loc[5],low.loc[95],color='k',alpha=0.2)
plt.plot(Int.columns.astype(int),Int.loc[50].values,c='mediumblue')
plt.fill_between(Int.columns.astype(int),Int.loc[17],Int.loc[83],color='blue',alpha=0.2)
plt.fill_between(Int.columns.astype(int),Int.loc[5],Int.loc[95],color='blue',alpha=0.2)
plt.plot(high.columns.astype(int),high.loc[50].values,c='darkorange')
plt.fill_between(high.columns.astype(int),high.loc[17],high.loc[83],color='orange',alpha=0.2)
plt.fill_between(high.columns.astype(int),high.loc[5],high.loc[95],color='orange',alpha=0.2)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: x/1000))  # convert y-axis to meters
plt.ylabel('SLR [m]')
plt.xlabel('Year')
plt.xticks(low.columns[::2].astype(int))
plt.xlim([2020,2100])
plt.ylim([0,2500])
median_handle = mlines.Line2D([], [], color='black', label='Median')
percentile_handle_17_83 = mpatches.Patch(color='black', alpha=0.4, label='66% Confidence Interval')
percentile_handle_5_95 = mpatches.Patch(color='black', alpha=0.2, label='90% Confidence Interval')
plt.legend(handles=[median_handle, percentile_handle_17_83, percentile_handle_5_95], bbox_to_anchor=(0, 0.9),loc='center left', fontsize='14')
plt.text(2090,350,'Low',color='black',fontsize=14,fontweight='bold')
plt.text(2083,970,'Intermediate',color='blue',fontsize=14,fontweight='bold')
plt.text(2069,1500,'High',color='orange',fontsize=14,fontweight='bold')
plt.savefig('figures/figure7.png', format='png', dpi=1000, bbox_inches='tight')

# calculate the EAL for each SLR scenario
low_loss = pd.DataFrame(func(low.values/1000, *popt), index=low.index, columns=low.columns)
Int_loss = pd.DataFrame(func(Int.values/1000, *popt), index=Int.index, columns=Int.columns)
high_loss = pd.DataFrame(func(high.values/1000, *popt), index=high.index, columns=high.columns)
# plot the EAL scenarios over time
plt.figure()
plt.plot([2020,2150],[portfolio_value,portfolio_value],'grey',linestyle='--')
plt.plot(low_loss.columns.astype(int),low_loss.loc[50],c='black',linestyle='-')
plt.fill_between(low_loss.columns.astype(int),low_loss.loc[17],low_loss.loc[83],color='black',alpha=0.2)
plt.fill_between(low_loss.columns.astype(int),low_loss.loc[5],low_loss.loc[95],color='black',alpha=0.2)
plt.plot(Int_loss.columns.astype(int),Int_loss.loc[50],c='blue',linestyle='-')
plt.fill_between(Int_loss.columns.astype(int),Int_loss.loc[17],Int_loss.loc[83],color='blue',alpha=0.2)
plt.fill_between(Int_loss.columns.astype(int),Int_loss.loc[5],Int_loss.loc[95],color='blue',alpha=0.2)
plt.plot(high_loss.columns.astype(int),high_loss.loc[50],c='orange',linestyle='-')
plt.fill_between(high_loss.columns.astype(int),high_loss.loc[17],high_loss.loc[83],color='orange',alpha=0.2)
plt.fill_between(high_loss.columns.astype(int),high_loss.loc[5],high_loss.loc[95],color='orange',alpha=0.2)
plt.yscale('log')
plt.ylabel('Average annual loss [USD]')
plt.xlabel('Year')
plt.xlim([2020,2100])
plt.ylim([1e8,1e10])
median_handle = mlines.Line2D([], [], color='black', linestyle='-', label='Median')
percentile_handle_17_83 = mpatches.Patch(color='black', alpha=0.4, label='66% Confidence Interval')
percentile_handle_5_95 = mpatches.Patch(color='black', alpha=0.2, label='90% Confidence Interval')
plt.legend(handles=[median_handle, percentile_handle_17_83, percentile_handle_5_95], bbox_to_anchor=(0, 0.75),loc='center left', fontsize='14')
plt.text(2025,portfolio_value*1.1,'Total portfolio value',color='grey',fontsize=18)
plt.text(2090,1.3e8,'Low',color='black',fontsize=14,fontweight='bold')
plt.text(2080,2.8e8,'Intermediate',color='blue',fontsize=14,fontweight='bold')
plt.text(2070,6e8,'High',color='orange',fontsize=14,fontweight='bold')
plt.savefig('figures/figure8.png', format='png', dpi=1000, bbox_inches='tight')

print('finished making all plots')
