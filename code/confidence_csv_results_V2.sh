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
#SBATCH --output=CONFIDENCE_PD_models_%j.log   # Standard output and error log
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

#OPTION 1: HPG KERNEL MODULE
#module load pytorch/1.13

# OPTION 2: SELF CREATED ENV
module load conda
conda activate /blue/ruogu.fang/charlietran/PD_Reproduction_V2/conda/envs/RetinaPD/
export PATH=/blue/ruogu.fang/charlietran/PD_Reproduction_V2/conda/envs/RetinaPD/bin:$PATH

cd /blue/ruogu.fang/charlietran/PD_Reproduction_V2/code/


# ML models
python confidence_csv_results_V2.py --model_name logistic_regression --experiment_tag overall  --project_dir ..
python confidence_csv_results_V2.py --model_name elastic_net --experiment_tag overall  --project_dir ..
python confidence_csv_results_V2.py --model_name svm_linear --experiment_tag overall  --project_dir ..
python confidence_csv_results_V2.py --model_name svm_rbf --experiment_tag overall  --project_dir ..
python confidence_csv_results_V2.py --model_name alexnet --experiment_tag overall  --project_dir ..
python confidence_csv_results_V2.py --model_name vgg --experiment_tag overall  --project_dir ..
python confidence_csv_results_V2.py --model_name googlenet --experiment_tag overall  --project_dir ..
python confidence_csv_results_V2.py --model_name inceptionv3 --experiment_tag overall  --project_dir ..
python confidence_csv_results_V2.py --model_name resnet --experiment_tag overall  --project_dir ..

python confidence_csv_results_V2.py --model_name logistic_regression --experiment_tag prevalent --project_dir ..
python confidence_csv_results_V2.py --model_name elastic_net --experiment_tag prevalent --project_dir ..
python confidence_csv_results_V2.py --model_name svm_linear --experiment_tag prevalent --project_dir ..
python confidence_csv_results_V2.py --model_name svm_rbf --experiment_tag prevalent --project_dir .. 
python confidence_csv_results_V2.py --model_name alexnet --experiment_tag prevalent --project_dir ..
python confidence_csv_results_V2.py --model_name vgg --experiment_tag prevalent --project_dir ..
python confidence_csv_results_V2.py --model_name googlenet --experiment_tag prevalent --project_dir ..
python confidence_csv_results_V2.py --model_name inceptionv3 --experiment_tag prevalent --project_dir ..
python confidence_csv_results_V2.py --model_name resnet --experiment_tag prevalent --project_dir ..


python confidence_csv_results_V2.py --model_name logistic_regression  --experiment_tag incident  --project_dir .. 
python confidence_csv_results_V2.py --model_name elastic_net --experiment_tag incident  --project_dir ..
python confidence_csv_results_V2.py --model_name svm_linear --experiment_tag incident  --project_dir ..
python confidence_csv_results_V2.py --model_name svm_rbf --experiment_tag incident  --project_dir ..
python confidence_csv_results_V2.py --model_name alexnet --experiment_tag incident  --project_dir ..
python confidence_csv_results_V2.py --model_name vgg --experiment_tag incident  --project_dir ..
python confidence_csv_results_V2.py --model_name googlenet --experiment_tag incident  --project_dir ..
python confidence_csv_results_V2.py --model_name inceptionv3 --experiment_tag incident  --project_dir ..
python confidence_csv_results_V2.py --model_name resnet  --experiment_tag incident  --project_dir ..

date



date
