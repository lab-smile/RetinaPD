# RetinaPD

This project is a SMILE Lab work for a paper entitled "Deep Learning Predicts Prevalent and Incident Parkinson's Disease From UK Biobank Fundus Imaging" in submission for publication. This work is supported by the National Science Foundation under Grant No. (NSF 2123809).  

The purpose of this project is for the binary classification of Parkinson's Disease from UK Biobank Fundus Imaging 

The work can currently be reproduced using UKB resources sourced under the privacy of the SMILE Lab. External reproduction cannot be accomplished without our permission for privacy reasons.

## 1. Packages

The project uses the following packages in general

- (Deep Learning) Pytorch 1.7.1 + Torchvision
- (Machine Learning) Scikit-Learn
- (Image Processing) CV2, Imageio, Pillow
- (Plotting) Seaborn, Matplotlib
- (XAI) Captum
- (CSV) Rapids CUDF (21.12) 

An environment can be installed with one of the following commands (mamba or conda). 

```
STEP 1.
- (OPTION 1. HPG-preferred) mamba create -p .../conda/envs/name_of_environment python=3.8
- (OPTION 2. Local/non-HPG) conda create -n name_of_environment python=3.8

STEP 2. Install Pytorch first
- (OPTION 1. HPG-preferred) mamba install pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch 
- (OPTION 2. Local/non-HPG) pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

STEP 3. Install remaining packages
pip install -r requirements_hpg.txt 
```
Install the packages found by online search using the traditional pip or conda install methods if an error is found.


## ** Guidelines ** 
Change all paths as necessary in the scripts hereon. If the script is not run in a shell script manner, the python script can still be made runnable (e.g., !python script.py --args) 

The shell scripts can be "sectioned" to run either all python experiments or be run multiple times partially. There is no requirement, except for the results plotting, for the shell scripts to run every line at once. 

## 2. Data pre-processing
Assuming the raw data has been acquired, the data can be split, resized, and allocated to folders by type. 
``
sbatch preprocess.py
``

The outputs will be stored in the data folder for Overall, Prevalent, and Incident data types. 

## 3. Train-ML models
Train (and test) machine learning models - SVM (RBF, Linear), Logistic Regression, and Elastic Net

``
sbatch train_ml.sh
``

The outputs will be stored as np arrays in the results folder for each evaluation metric.

## 4. Train-DL models
Train deep learning models - AlexNet, VGG16, ResNet50, GoogleNet, InceptionV3

``
sbatch train_dl.sh
``

The outputs will be stored as Pytorch weight models in the models folder.

## 5. Test-DL models
Evaluate each of the deep learning models at test-time.
``
sbatch test_dl.sh
``

The outputs are stored as np arrays in the results folder for each evaluation metric. 

## 6. XAI infidelity and sensitivity
Estimate the explanation infidelity and sensitivity measures for model robustness evaluation.
``
sbatch XAI_metrics_test.sh
``

The outputs will be stored in the results folder under each model and data-type. 

## 7. Results analysis

``
Step 1. Confidence Intervals
sbatch confidence_csv_results.sh	

Step 2. Plots
sbatch results_plot.sh
``

The outputs will be stored in the results folder including the csv and pdf's of the plots. 

## 8. Population Characteristics

Explore the notebook for the dataframes to understand how the subject characteristics and statistics were analyzed.

``
CSV_Population_Characteristics.ipynb
``


