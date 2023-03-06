#!/bin/bash
#SBATCH --job-name=preprocess_PD    # Job name
#SBATCH --mail-type=End,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=charlietran@ufl.edu    # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=2                    # Run on a single CPU
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=25gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=6:00:00               # Time limit hrs:min:sec
#SBATCH --output=preprocess_PD_%j.log   # Standard output and error log
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang



#OPTION 1: HPG KERNEL MODULE
module load pytorch/1.7.1

# OPTION 2: SELF CREATED ENV
#module load conda
#conda activate /blue/ruogu.fang/charlietran/PD_Reproduction_V2/RetinaPD
#export PATH=/blue/ruogu.fang/charlietran/PD_Reproduction_V2/RetinaPD/bin:$PATH

cd /blue/ruogu.fang/charlietran/PD_Reproduction_V2/code/


# We do not explicily need Pytorch but the PyTorch Environment Module from HPG contains a lot of good packages.
python preprocess.py --project_dir ..



date
