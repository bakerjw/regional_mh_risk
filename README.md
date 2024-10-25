# regional_mh_risk
[![DOI](https://zenodo.org/badge/878637905.svg)](https://doi.org/10.5281/zenodo.13994340)
This project contains python functions to run a regional multi-hazard risk assessment under climate change. This code recreates figures 
from Mongold, E. and Baker, J.W. (Forthcoming). The building inventory values are set to the average for privacy, so some results may not 
align exactly with the paper. 

This package is divided by the hazard considered. The currently available hazards are:
1. Earthquake (including liquefaction)
2. Coastal flooding
3. Tsunami

To re-create the environment used for this project on anaconda: 
conda env create -f environment.yml

To import the regional_mh_risk package, you should navigate to the folder /regional_mh_risk/ which contains setup.py, and in the conda environment, run:
python setup.py install

To run the example from start to finish, follow these steps:
0. Setup the building inventory (filling tax assessor data with NSI data, using building_inventory.py)
1. Run the earthquake ground motions: example/ground_motions/run_pypsha_clean.ipynb
2. Run the liquefaction output*: setup_run.py and flex_liq_run.py (run using slurm_array.sh*), then earthquake_postprocess.py
3. Run coastal flooding and tsunami hazards: cf_run.py and tsu_run.py
4. Once all hazards are run, risk_run.py outputs many of the risk metric figures. 
5. In parallel, adapt_raise.py, adapt_retreat.py, and adapt_retrofit.py* can all be run. 
6. To postprocess the adaptations, run adapt_compare.py

* These steps were run on remote clusters, and may be difficult to run on the whole dataset on a personal computer. 
