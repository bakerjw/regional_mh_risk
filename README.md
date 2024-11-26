# regional_mh_risk
This project contains python functions to run a regional multi-hazard risk assessment under climate change. This code recreates figures 
from Mongold, E. and Baker, J.W. (Forthcoming). The building inventory values are set to the average for privacy, so some results may not 
align exactly with the paper. 

This package is divided by the hazard considered. The currently available hazards are:
1. Earthquake (including liquefaction)
2. Coastal flooding
3. Tsunami

The risk analysis is performed on residential buildings, with the fragility functions built into each hazard. Additional files are contained as wrappers to perform the following:
1. Pre-processing
2. Simulations
3. Post-processing

To re-create the environment used for this project on anaconda: 
conda env create -f environment.yml

To import the regional_mh_risk package, you should navigate to the folder /regional_mh_risk/ which contains setup.py, and in the conda environment, run:
python setup.py install


To run the example from start to finish, follow these steps:
First, some necessary downloads:
Alameda_tsunami_tifs from the bottom of this webpage: https://www.conservation.ca.gov/cgs/tsunami/reports
coastal_flood_rasters from this webpage: https://explorer.adaptingtorisingtides.org/download


0. Setup the building inventory (filling tax assessor data with NSI data, using building_inventory.py)
1. Run the earthquake ground motions: example/ground_motions/run_pypsha_clean.ipynb
2. Run the liquefaction output*: setup_run.py and flex_liq_run.py (run using slurm_array.sh*), then earthquake_postprocess.py
3. Run coastal flooding and tsunami hazards: cf_run.py and tsu_run.py
4. Once all hazards are run, risk_run.py outputs many of the risk metric figures. 
5. In parallel, adapt_raise.py, adapt_retreat.py, and adapt_retrofit.py* can all be run. 
6. To postprocess the adaptations, run adapt_compare.py

* These steps were run on remote clusters, and may be difficult to run on the whole dataset on a personal computer. 