#!/bin/bash
#SBATCH --job-name=TEST_PD_models    # Job name
#SBATCH --mail-type=End,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=charlietran@ufl.edu    # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=25gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=2:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --output=TEST_DL_PD_models_%j.log   # Standard output and error log
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang


#OPTION 1: HPG KERNEL MODULE
#module load pytorch/1.13

# OPTION 2: SELF CREATED ENV
module load conda
conda activate /blue/ruogu.fang/charlietran/conda/envs/RetinaPD/
export PATH=/blue/ruogu.fang/charlietran/conda/envs/RetinaPD/bin:$PATH

cd /blue/ruogu.fang/charlietran/PD_Reproduction_V4/code/

python test_dl_paths.py --model_name alexnet --experiment_tag overall  --project_dir ..
python test_dl_paths.py --model_name alexnet --experiment_tag prevalent --project_dir ..
python test_dl_paths.py --model_name alexnet --experiment_tag incident  --project_dir ..



date
