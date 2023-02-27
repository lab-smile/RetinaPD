import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display


import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import random_projection

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

import cv2
import time


def load_images(fold_path):
    files = os.listdir(fold_path)

    cn_path = fold_path + '/HC/'
    pd_path = fold_path + '/PD/'
    cn_image_path = os.listdir(cn_path)
    pd_image_path = os.listdir(pd_path)
    cn_image = plt.imread(os.path.join(cn_path,cn_image_path[0]))
    cn_image = cv2.resize(cn_image, (256,256))
    m = len(cn_image)

    cn_image_resize = np.reshape(cn_image, (1,m*m*3))
    cn = cn_image_resize
    
    pd_image = plt.imread(os.path.join(pd_path,pd_image_path[0]))
    pd_image = cv2.resize(pd_image, (256,256))
    pd_image_resize = np.reshape(pd_image, (1,m*m*3))
    pd = pd_image_resize
    
    
    label = np.concatenate((np.zeros(len(cn_image_path)),np.ones(len(pd_image_path))))
    for i in range(1, len(cn_image_path)):
        #print(i)
        cn_image = plt.imread(os.path.join(cn_path,cn_image_path[i]))
        cn_image = cv2.resize(cn_image, (256,256))
        cn_image_resize = np.reshape(cn_image, (1,m*m*3))
        
        cn = np.concatenate((cn, cn_image_resize), axis = 0)
    
    for i in range(1, len(pd_image_path)):
         pd_image = plt.imread(os.path.join(pd_path,pd_image_path[i]))
         pd_image = cv2.resize(pd_image, (256,256))
         pd_image_resize = np.reshape(pd_image, (1,m*m*3))
         pd = np.concatenate((pd, pd_image_resize), axis = 0)
    
    dataset = np.concatenate((cn,pd),axis = 0)
    
    
    return dataset, label 



def main():
   # files = os.listdir(path)
    #test_path = path + '/test/'
   # train_path = path + '/train/'


    X_train, y_train = load_images(train_path)
    X_test, y_test = load_images(test_path)
    #print('len of train set', X_train.shape, 'len of test set', X_test.shape)

    scaler = StandardScaler()
    scaler.fit(X_train)
    # I actually have no idea if you need to use X_train = , or just use scaler.fit(X_train).
    # just do this I guess

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #scaler.transform(X_train)

    dtc = SVC(kernel='linear', probability=True, random_state=42)
    dtc.fit(X_train, y_train)
    
    if model_name == 'svm_linear':
        model = SVC(kernel  = 'linear', probability = True, random_state = 42)
    if model_name == 'svm_rbf':
        model = SVC(kernel  = 'rbf', probability = True, random_state = 42)
    if model_name == 'logistic_regression':
        model = LogisticRegression()
    if model_name == 'elastic_net':
        model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5)
    model.fit(X_train, y_train)
    #dtc = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5)
    #dtc.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)
    y_class = np.argmax(y_pred, axis = 1)
    # y_class = np.rint(y_pred)
    report = classification_report(y_test, y_class, digits = 3, output_dict = True)
    #calculate auc
    auc_test = roc_auc_score(y_test, y_pred[:, 1])
    auc_scores.append(auc_test)
    accuracy_scores.append(report['accuracy'])
    ppv_scores.append(report['1.0']['precision'])
    npv_scores.append(report['0.0']['precision'])
    sensitivity_scores.append(report['1.0']['recall'])
    specificity_scores.append(report['0.0']['recall'])
    f1_scores.append(report['1.0']['f1-score'])


    # append the class predictions
    prob_outputs.append(y_pred)
    parkinson_probability.append(y_pred[:, 1])
    class_predictions.append(y_class)
    ground_truth.append(y_test)
    
    
    
if __name__ == '__main__':

# Not sure if I actually need this but store it just incase


    prob_outputs = [] 
    parkinson_probability = [] 
    class_predictions = []
    ground_truth = [] 

    auc_scores = [] 
    accuracy_scores = []
    ppv_scores = [] 
    npv_scores = [] 
    sensitivity_scores = [] 
    specificity_scores = [] 
    f1_scores = [] 

    print('Code start')
    parser = argparse.ArgumentParser(description = 'What the program does')
    parser.add_argument('--model_name', type = str)
    parser.add_argument('--experiment_tag', type = str)
    parser.add_argument('--project_dir', type = str)
    args = parser.parse_args()

    ### You need to change the base data_folder based on the eperiment (Overall, Prevalent, Incident)
    # Your options are [svm_linear, svm_rbf, logistic_regression, elastic_net]
    model_name = args.model_name
    base_folder = args.project_dir
    experiment_tag = args.experiment_tag
    
    ########################

    if experiment_tag == 'overall': 
        data_extension = 'Overall_KFold_Data'
    if experiment_tag == 'prevalent':
        data_extension = 'Prevalent_KFold_Data'
    if experiment_tag == 'incident':
        data_extension = 'Incident_KFold_Data'      
    

    for j in range(1,6):
        for k in range(1,6):
            time1 = time.time()
            train_path = os.path.join(base_folder, 'data',  data_extension, 'R' + str(j),  str(k) , 'train')
            test_path =  os.path.join(base_folder, 'data', data_extension, 'R' + str(j), str(k) , 'test')
            main()
            time2 = time.time()
            print('Time (minutes) to Run CV:', (time2 - time1) / 60)
    print('---------------------------------------------------------------')
    print('AVERAGE SCORE REPORT')
    print('---------------------------------------------------------------')
    print('AVERAGE AUC', '{:.2f}'.format(np.average(auc_scores)),'({:.2f})'.format(np.std(auc_scores)))
    print('AVERAGE Accuracy', '{:.2f}'.format(np.average(accuracy_scores)),'({:.2f})'.format(np.std(accuracy_scores)))
    print('AVERAGE PPV', '{:.2f}'.format(np.average(ppv_scores)),'({:.2f})'.format(np.std(ppv_scores)))
    print('AVERAGE NPV', '{:.2f}'.format(np.average(npv_scores)), '({:.2f})'.format(np.std(npv_scores)))
    print('AVERAGE Sensitivity', '{:.2f}'.format(np.average(sensitivity_scores)), '({:.2f})'.format(np.std(sensitivity_scores)))
    print('AVERAGE Specificity', '{:.2f}'.format(np.average(specificity_scores)), '({:.2f})'.format(np.std(specificity_scores)))
    print('AVERAGE F1 scores', '{:.2f}'.format(np.average(f1_scores)),'({:.2f})'.format(np.std(f1_scores)))
    print('---------------------------------------------------------------')



    output_base_dir_name = os.path.join(base_folder, 'results', experiment_tag, model_name)
    os.makedirs(output_base_dir_name, exist_ok = True)

    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'auc_scores.npy'), auc_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'accuracy_scores.npy'), accuracy_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'ppv_scores.npy'), ppv_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'npv_score.npy'), npv_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'sensitivity_scores.npy'), sensitivity_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'specificity_scores.npy'), specificity_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'f1_scores.npy'), f1_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'pd_probability.npy'), parkinson_probability)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'class_predictions.npy'), class_predictions)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'ground_truth.npy'), ground_truth)


    # print statements
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'auc_scores.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'accuracy_scores.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name +  '_' + 'ppv_scores.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'npv_scores.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'sensitivity_scores.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'specificity_scores.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'prob_outputs.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'pd_probability.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'class_predictions.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'f1_scores.npy'))

  