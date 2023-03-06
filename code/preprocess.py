import os 
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import matplotlib.pyplot as plt
import imageio
import cv2
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)
import argparse
import time

parser = argparse.ArgumentParser(description = 'What the program does')
parser.add_argument('--project_directory', type = str)

def data_splitter(input_path,
                  PD_tag, 
                  HC_tag, 
                  output_folder,
                  seed_number):
  path = input_path
  files = os.listdir(path)
  hc_path = os.path.join(path, PD_tag)
  pd_path = os.path.join(path, HC_tag)
  
  N = 5 
  ids = [] 
  for img in os.listdir(pd_path):
      ids.append(img[0:7] + '_PD')
  for img in os.listdir(hc_path):
      ids.append(img[0:7] + '_CN')
  unique_ids = np.unique(ids) #
  
  labels = [] 
  
  for id in unique_ids:
      if id[-2:] == 'CN':
          labels.append(0)
      elif id[-2:] == 'PD':
          labels.append(1)
  
  labels = np.array(labels)
  count = 0
  dim = (512,512)
  kf_outer = StratifiedKFold(n_splits= 5, random_state = seed_number, shuffle = True)  
  for train_index, test_index in kf_outer.split(unique_ids, labels):
      train_ids, test_ids = unique_ids[train_index], unique_ids[test_index]
      train_lbls, test_lbls = labels[train_index], labels[test_index]  
  
      cn_train_list = [] 
      ad_train_list = []
  
      count += 1
      #print(pd_path)
      save_path = os.path.join(output_folder, str(count))
      #save_path = os.path.join(path, str(count)) 
  
  
      os.makedirs(os.path.join(save_path, 'train'), exist_ok = True) # create train file
  
      os.makedirs(os.path.join(save_path, 'test'), exist_ok = True) # create test file
      save_pd_train_path = os.path.join(save_path, 'train','PD')
      os.makedirs(save_pd_train_path, exist_ok = True)    # create tran/PD/
      save_hc_train_path = os.path.join(save_path, 'train', 'HC')
      os.makedirs(save_hc_train_path, exist_ok = True) # create /train/HC/
  
      for eid in train_ids:
  
          #split images to /train/PD/
          if eid[-2:] == 'PD':
              for img in os.listdir(pd_path):
                  if str(eid[0:7]) in img:
                      image =  plt.imread(os.path.join(pd_path,img))           
                      image =cv2.resize(image, dim)
                      #print(os.path.join(save_pd_train_path,img))
                      imageio.imwrite(os.path.join(save_pd_train_path,img), image)
  
          # split images to /train/HC/
          if eid[-2:] == 'CN':
              for img in os.listdir(hc_path):
                  if str(eid[0:7]) in img:
                      image =  plt.imread(os.path.join(hc_path,img))
                      image =cv2.resize(image, dim)
                      imageio.imwrite(os.path.join(save_hc_train_path,img), image)
  
      save_pd_test_path = os.path.join(save_path, 'test','PD')
      os.makedirs(save_pd_test_path, exist_ok = True) # create /test/PD/
      save_hc_test_path = os.path.join(save_path, 'test', 'HC')
      os.makedirs(save_hc_test_path, exist_ok = True) # create /test/HC/
  
      for eid in test_ids:
          #split images to /test/PD/
          if eid[-2:] == 'PD':
              for img in os.listdir(pd_path):
                  if str(eid[0:7]) in img:
                      image =  plt.imread(os.path.join(pd_path,img))     
                      image =cv2.resize(image, dim)
                      #print(os.path.join(save_pd_test_path,img))
                      imageio.imwrite(os.path.join(save_pd_test_path,img), image)
          #split images to /test/PD/
          if eid[-2:] == 'CN':
              for img in os.listdir(hc_path):
                  if str(eid[0:7]) in img:
                      image =  plt.imread(os.path.join(hc_path,img))
                      image = cv2.resize(image, dim)
                      imageio.imwrite(os.path.join(save_hc_test_path,img), image)
                        

#self reminder I used this link: https://stackoverflow.com/questions/52165705/how-to-ignore-root-warnings 
# to figure out how to remove the warnings

#You need only to change to your project directory here 

def main():

##### OVERALL ########## 
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD', 
                HC_tag = 'HC',
                output_folder = os.path.join(project_directory, 'data/Overall_KFold_Data/R1/'),
                seed_number = 5) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD', 
                HC_tag = 'HC',              
                output_folder = os.path.join(project_directory, 'data/Overall_KFold_Data/R2/'),
                seed_number = 25) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD', 
                HC_tag = 'HC',
                output_folder = os.path.join(project_directory,'data/Overall_KFold_Data/R3/'),
                seed_number = 50) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD', 
                HC_tag = 'HC',
                output_folder = os.path.join(project_directory, 'data/Overall_KFold_Data/R4/'),
                seed_number = 75) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD', 
                HC_tag = 'HC',
                output_folder = os.path.join(project_directory, 'data/Overall_KFold_Data/R5/'),
                seed_number = 100)  
                
                
##### PREVALENT ########## 
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'),  
                PD_tag = 'PD_prevalent', 
                HC_tag = 'PD_prevalent_hc',
                output_folder = os.path.join(project_directory, 'data/Prevalent_KFold_Data/R1/'),
                seed_number = 5) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'),  
                PD_tag = 'PD_prevalent', 
                HC_tag = 'PD_prevalent_hc',           
                output_folder = os.path.join(project_directory, 'data/Prevalent_KFold_Data/R2/'),
                seed_number = 25) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'),  
                PD_tag = 'PD_prevalent', 
                HC_tag = 'PD_prevalent_hc',
                output_folder = os.path.join(project_directory, 'data/Prevalent_KFold_Data/R3/'),
                seed_number = 50) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'),  
                PD_tag = 'PD_prevalent', 
                HC_tag = 'PD_prevalent_hc',
                output_folder = os.path.join(project_directory, 'data/Prevalent_KFold_Data/R4/'),
                seed_number = 75)  
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD_prevalent', 
                HC_tag = 'PD_prevalent_hc',
                output_folder = os.path.join(project_directory, 'data/Prevalent_KFold_Data/R5/'),
                seed_number = 100) 
  
##### INCIDENT #########

  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'),
                PD_tag = 'PD_incident', 
                HC_tag = 'PD_incident_hc',
                output_folder = os.path.join(project_directory,'data/Incident_KFold_Data/R1/'),
                seed_number = 5) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD_incident', 
                HC_tag = 'PD_incident_hc',           
                output_folder = os.path.join(project_directory, 'data/Incident_KFold_Data/R2/'),
                seed_number = 25) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD_incident', 
                HC_tag = 'PD_incident_hc',
                output_folder = os.path.join(project_directory, 'data/Incident_KFold_Data/R3/'),
                seed_number = 50) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'),
                PD_tag = 'PD_incident', 
                HC_tag = 'PD_incident_hc',
                output_folder = os.path.join(project_directory, 'data/Incident_KFold_Data/R4/'),
                seed_number = 75) 
  
  data_splitter(input_path = os.path.join(project_directory, 'data/Raw_Data/'), 
                PD_tag = 'PD_incident', 
                HC_tag = 'PD_incident_hc',
                output_folder = os.path.join(project_directory, 'data/Incident_KFold_Data/R5/'),
                seed_number = 100)

if __name__ == '__main__':
  time1 = time.time()
  args = parser.parse_args()
  project_directory = args.project_directory
  main()
  time2 = time.time()
  print((time2 - time1)/60, 'minutes')