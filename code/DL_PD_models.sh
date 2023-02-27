#!/bin/bash
#SBATCH --job-name=PD_models    # Job name
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
#SBATCH --output=PD_models_%j.log   # Standard output and error log
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang



module load pytorch/1.7.1

cd /blue/ruogu.fang/charlietran/PD_Reproduction/code/

python train_dl.py --model_name alexnet --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py  --model_name alexnet --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py --model_name alexnet --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5

python train_dl.py  --model_name vgg --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py  --model_name vgg --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py  --model_name vgg --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5

python train_dl.py --model_name resnet --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py --model_name resnet --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py --model_name resnet  --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5

python train_dl.py --model_name googlenet --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py --model_name googlenet --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py --model_name googlenet --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5

python train_dl.py --model_name inceptionv3 --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py --model_name inceptionv3 --experiment_tag prevalent --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5
python train_dl.py --model_name inceptionv3 --experiment_tag incident  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 5


date
