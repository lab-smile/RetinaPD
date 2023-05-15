#!/bin/bash
#SBATCH --job-name=PD_models    # Job name
#SBATCH --mail-type=End,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=charlietran@ufl.edu    # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=10gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --output=PLOTTING_models_%j.log   # Standard output and error log
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang


#OPTION 1: HPG KERNEL MODULE
#module load pytorch/1.7.1

# OPTION 2: SELF CREATED ENV
module load conda
conda activate /blue/ruogu.fang/charlietran/conda/envs/RetinaPD/
export PATH=/blue/ruogu.fang/charlietran/conda/envs/RetinaPD/bin:$PATH

cd /blue/ruogu.fang/charlietran/PD_Reproduction_V4/code/


# The results plots were split across scripts because
# one for organization
# two because matplotlib does not cooperate well with figure environments unless coded properly
#python acc_results_plots.py --experiment_tag overall  --project_dir ..
#python auc_results_plots.py --experiment_tag overall  --project_dir ..
#python sensitivity_results_plots.py --experiment_tag overall  --project_dir ..
#python infidelity_results_plots.py --experiment_tag overall --project_dir ..

#python acc_results_plots.py --experiment_tag prevalent  --project_dir ..
#python auc_results_plots.py --experiment_tag prevalent  --project_dir ..
#python sensitivity_results_plots.py --experiment_tag prevalent  --project_dir ..
#python infidelity_results_plots.py --experiment_tag prevalent --project_dir ..

#python acc_results_plots.py --experiment_tag incident  --project_dir ..
#python auc_results_plots.py --experiment_tag incident  --project_dir ..
python sensitivity_results_plots.py --project_dir ..
python infidelity_results_plots.py --project_dir ..

date
