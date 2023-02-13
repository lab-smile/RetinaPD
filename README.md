# RetinaPD

This project is a SMILE Lab work for a paper enitled "Deep Learning Predicts Prevalent and Inciden Parkinson's Disease From UK Biobank Fundus Imaging" in submission to JAMA Open Network. This work is supported by the National Science Foundation under Grant No. (NSF 2123809). 

The purpose of this project is for the binary classification of Parkinson's Disease from UK Biobank Fundus Imaging 

The work can currently be reproduced using UKB resources sourced under the privacy of the SMILE Lab. External reproduction cannot be accomplished without our permission for privacy reasons.

# 1. Packages

The environment is pre-established in HiPerGator using PyTorch 1.7.1 (training and evaluation) and RAPIDS 21.12 (CuDF) 

# 2. Project Pipeline 

The project pipeline of data collection, processing, training, and evaluation can be found in the jupyter notebook. The training and evaluation will consist of coventional machine learning models (SVM, Logistic Regression, ElasicNet) and several deep learning models (AlexNet, VGG16, GoogleNet, ResNet50, and Inception-V3) under five fold cross validation.  

```
RetinaPD_Project_Code.ipynb 
```

# 3. The subject characteristics and their statistical comparisons from the UK Biobank can be found in the jupyter notebook 

```
CSV_Population_Characteristics.ipynb
```

