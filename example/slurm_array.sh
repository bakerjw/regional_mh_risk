#!/bin/bash
#
#BATCH --job-name = liqrun
#
#SBATCH --time=00:50:00
#SBATCH --ntasks-per-node=1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-11

module load viz
module load python/3.9.0

echo This is task $SLURM_ARRAY_TASK_ID

python3 'flex_liq_run.py' $SLURM_ARRAY_TASK_ID 0.0

