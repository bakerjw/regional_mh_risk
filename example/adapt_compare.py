## adapt_compare.py to compare the adaptation strategies run for Alameda
## By Emily Mongold 08/13/2024

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the y tick labels
plt.rc('legend', fontsize=12)  # fontsize of the legend
plt.rcParams["figure.figsize"] = (10, 7)
orig_cmap = plt.cm.GnBu
cols = orig_cmap(np.linspace(1, 0.3, 10))
cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", cols)
# define four colors by the hex codes blues and greens
colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']

## load the values for mitigation comparison
with open('adaptation_raised.pkl','rb') as f:
    raised = pickle.load(f)

with open('adaptation_retreat_norm.pkl','rb') as f:
    retreat = pickle.load(f)

with open('adaptation_retrofit_both.pkl','rb') as f:
    retrofit = pickle.load(f)

print('Inputs loaded')

adapt_bldgs = pd.read_csv('eq_adapt_bldgs.csv')
total_inventory = adapt_bldgs['ImprovementValue'].sum()
retreat_inventory = adapt_bldgs['ImprovementValue'][adapt_bldgs['retreat']==0].sum()
lost_inventory = adapt_bldgs['ImprovementValue'][adapt_bldgs['retreat']==1].sum()

slr_levels = [str(x) for x in retreat['baseline'].keys()]
eal = list(retrofit['baseline_eal'].values())
eal_present = list(retreat['present_retreat'].values())
eal_slr = list(retreat['slr_retreat'].values())
eal_1ft = list(raised['raised_1ft'].values())
eal_3ft = list(raised['raised_3ft'].values())
eal_retrofit = list(retrofit['retrofit40pct'].values())

# calculate the difference in EAL between the baseline and each mitigation strategy for each SLR level
diff_present = np.array([b-e for b, e in zip(eal, eal_present)])/1e6
diff_slr =  np.array([b-e for b, e in zip(eal, eal_slr)])/1e6
diff_1ft =  np.array([b-e for b, e in zip(eal, eal_1ft)])/1e6
diff_3ft =  np.array([b-e for b, e in zip(eal, eal_3ft)])/1e6
diff_retrofit =  np.array([b-e for b, e in zip(eal, eal_retrofit)])/1e6

# make a bar plot of the differences
fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.15
for i, bar in enumerate(slr_levels):
    ax.bar(i - 2*bar_width, diff_present[i], color=colors[0], width=bar_width,edgecolor='k')
    ax.bar(i - bar_width, diff_slr[i], color=colors[1], width=bar_width,edgecolor='k')
    ax.bar(i , diff_1ft[i], color=colors[2], width=bar_width,edgecolor='k')
    ax.bar(i + bar_width, diff_3ft[i], color=colors[3], width=bar_width,edgecolor='k')
    ax.bar(i + 2*bar_width, diff_retrofit[i], color='purple', width=bar_width,edgecolor='k')
plt.legend(['Retreat Present-day','Retreat SLR risk', 'Raised 1ft','Raised 3ft', 'Retrofit'])
ax.set_xlabel('Sea Level Rise (SLR) Amount [m]')
ax.set_ylabel('Reduction in $AAL$ [Million USD]')
plt.yscale('log')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val)}'))
plt.xticks(range(len(slr_levels)), slr_levels)
plt.tight_layout()
plt.savefig('figures/figure10.png', format='png', dpi=1000, bbox_inches='tight')

### load exceedance curves from raised, retreat, and retrofit
exceed_baseline = retrofit['exceedance_base']
exceed_retreat_present = retreat['exceedance_retreat_pres']
exceed_retreat_slr = retreat['exceedance_retreat_slr']
exceed_raise_1ft = raised['exceedance_raised1ft']
exceed_raise_3ft = raised['exceedance_raised3ft']
exceed_retrofit = retrofit['exceedance_retrofit']

## plot the change in exceedance curves between adaptation strategies
losses = np.arange(0, 5e9, 5e5) # total portfolio is 7e9

# make a plot with different RPs along the x axis and the same metric of loss reduction on the y axis
rp_list = [5,10,50,200,500,1000]
num=0
fignum=['a','b']
for slr in [0.0,0.75]:
    rp500 = [exceed_baseline[slr].index[max(max(np.where(exceed_baseline[slr]>(1/rp))))] for rp in rp_list]
    rp500_present = [exceed_retreat_present[slr].index[max(max(np.where(exceed_retreat_present[slr]>(1/rp))))] for rp in rp_list]
    rp500_slr = [exceed_retreat_slr[slr].index[max(max(np.where(exceed_retreat_slr[slr]>(1/rp))))] for rp in rp_list]
    rp500_1ft = [exceed_raise_1ft[slr].index[max(max(np.where(exceed_raise_1ft[slr]>(1/rp))))] for rp in rp_list]
    rp500_3ft = [exceed_raise_3ft[slr].index[max(max(np.where(exceed_raise_3ft[slr]>(1/rp))))] for rp in rp_list]
    rp500_retrofit = [exceed_retrofit[slr].index[max(max(np.where(exceed_retrofit[slr]>(1/rp))))] for rp in rp_list]

    diff_present = np.array([b-e for b, e in zip(rp500, rp500_present)])/1e6
    diff_slr = np.array([b-e for b, e in zip(rp500, rp500_slr)])/1e6
    diff_1ft = np.array([b-e for b, e in zip(rp500, rp500_1ft)])/1e6
    diff_3ft = np.array([b-e for b, e in zip(rp500, rp500_3ft)])/1e6
    diff_retrofit = np.array([b-e for b, e in zip(rp500, rp500_retrofit)])/1e6
    # make a bar plot of the differences
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.15
    for i, bar in enumerate(rp_list):
        ax.bar(i - 2*bar_width, diff_present[i], color=colors[0], width=bar_width,edgecolor='k')
        ax.bar(i - bar_width, diff_slr[i], color=colors[1], width=bar_width,edgecolor='k')
        ax.bar(i , diff_1ft[i], color=colors[2], width=bar_width,edgecolor='k')
        ax.bar(i + bar_width, diff_3ft[i], color=colors[3], width=bar_width,edgecolor='k')
        ax.bar(i + 2*bar_width, diff_retrofit[i], color='purple', width=bar_width,edgecolor='k')
    plt.legend(['Retreat Present-day','Retreat SLR risk', 'Raised 1ft','Raised 3ft', 'Retrofit'])
    # Set the labels and title
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('Reduction in loss, $\Delta c(\lambda)$ [Million USD]')
    plt.xticks(range(len(rp_list)), rp_list)
    plt.tight_layout()
    plt.savefig('figures/figure11'+fignum[num]+'.png', format='png', dpi=1000, bbox_inches='tight')
    num+=1

print('ran and saved figure')