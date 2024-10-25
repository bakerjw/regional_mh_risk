import numpy as np
import pickle
from regional_mh_risk.simple_liquefaction import setup_cpt, assign_gwt

print('Imported packages')

inputdir = './ground_motions/USGS_CPT_data/'
cpts = setup_cpt(inputdir)
cpts = assign_gwt(cpts, './gw_tifs/')

print('Set up cpts')

with open('./base_cpts.pkl','wb') as f:
    pickle.dump(cpts,f,protocol=pickle.HIGHEST_PROTOCOL)

nsim = 2423
fun = np.random.randint(2, size=nsim)
C_FC = np.random.uniform(-0.2, 0.2, nsim)
np.savetxt('./fun.csv', fun, delimiter=',')
np.savetxt('./cfc.csv', C_FC, delimiter=',')

print('saved')
