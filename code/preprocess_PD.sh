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
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --output=preprocess_PD_%j.log   # Standard output and error log
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang



module load pytorch/1.7.1

cd /blue/ruogu.fang/charlietran/PD_Reproduction/code/


## This code technically does not need GPU. It will be here recklessly as I simply copied over the slurm script. 
## Also does not really take 6 hours, maybe like an hour? 
# I load module load pytorch.... for no good reason really except that it contains all of the basic packages that I need from sklearn
python preprocess.py --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 



date
