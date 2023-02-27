# RetinaPD

This project is a SMILE Lab work for a paper entitled "Deep Learning Predicts Prevalent and Incident Parkinson's Disease From UK Biobank Fundus Imaging" in submission for publication. This work is supported by the National Science Foundation under Grant No. (NSF 2123809). 

This research conducts a comprehensive evaluation of deep learning and machine learning models of Parkinson's disease from UK Biobank Fundus Imaging. 


# 1. Packages

The environment has been tested in HiPerGator using PyTorch 1.7.1 (training and evaluation) and RAPIDS 21.12 (CuDF) 

# 2. Preprocessing (Data acquisition, resizing, and splitting)

## 2.1 (Jupyter Version) Run sections 1-3 in the jupyter notebook: RetinaPD_Project_Code.ipynb 

## 2.2 (Terminal Version) Run the following commands in terminal to copy the data. Edit the parameters in the slurm script as necessary (email, log outputs, etc.) 

```
cp /blue/ruogu.fang/charlietran/PD_Reproduction/data/PD_raw_data.zip -d your_data_blue_project_folder
unzip -q your_data_blue_project_folder/PD_raw_data.zip -d your_data_blue_project_folder/data/
```

Then, run the preprocess.py with the provided slurm script in the terminal. Edit the parameters in the slurm script as necessary (email, log outputs, etc.) 

```
sbatch pre_process_PD.sh 
(EDIT:) python preprocess.py --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
```

# 3. Model training and Evaluation

## 3.1 (Jupyter Version) Run sections 4 and 5 for the deep learning and machine learning evaluation in the jupyter notebook: RetinaPD_Project_Code.ipynb
## 3.2 (Terminal Version) Run the deep learning and machine learning models using slurm 

For the deep learning models, change the following arguments below according to the python Parser. The official paper used 100 epochs with early stopping. The available models are alexnet, vgg, resnet, googlenet, and inceptionv3. The term experiment tag refers to overall, prevalent, or incident. 
```
sbatch DL_PD_models.sh
(EDIT:) python train_dl.py --model_name alexnet --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ --epochs 100
```
For the machine learning models, change the following arguments below according to the python Parser. The avaukabke nideks are svm_rbf, svm_linear, logistic_regression, and elastic_net. The term experiment tag refers to overall, prevalent, or incident. 
```
sbatch ML_PD_models.sh
(EDIT:) python train_ml.py --model_name svm_rbf --experiment_tag overall  --project_dir /blue/ruogu.fang/charlietran/PD_Reproduction/ 
```

# 4. Test-time Explanation and Plotting

Run the remaining portion of the jupyter notebook: RetinaPD_Project_Code.ipynb, Sections 6 and 7. The numpy arrays of all results is located in the results folder as a zip file. This can be used in conjunction with the RetinaPD code for the plotting.

# 5. Subject Clinical Measures

The subject characteristics and their statistical comparisons from the UK Biobank can be found in the jupyter notebook using the Rapids (21.12) CuDF library. 

```
CSV_Population_Characteristics.ipynb
```





