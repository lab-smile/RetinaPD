#!/bin/bash
#SBATCH --job-name=ML_PD_models  # Job name
#SBATCH --mail-type=End,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=charlietran@ufl.edu    # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=2                    # Run on a single CPU
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=25gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --output=ML_PD_models_%j.log   # Standard output and error log
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang



module load pytorch/1.7.1

cd /blue/ruogu.fang/charlietran/PD_Reproduction/code/

#This is similar to the DL models sh file but there are no epochs for ML models
# I load module load pytorch.... for no good reason really except that it contains all of the basic packages that I need from sklearn
python train_ml.py --model_name svm_rbf --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
python train_ml.py --model_name svm_rbf --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
python train_ml.py --model_name svm_rbf --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 

python train_ml.py --model_name svm_linear --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
python train_ml.py --model_name svm_linear --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
python train_ml.py --model_name svm_linear --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 

python train_ml.py --model_name logistic_regression --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/
python train_ml.py --model_name logistic_regression --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
python train_ml.py --model_name logistic_regression  --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 

python train_ml.py --model_name elastic_net --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
python train_ml.py --model_name elastic_net --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
python train_ml.py --model_name elastic_net --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 


date
