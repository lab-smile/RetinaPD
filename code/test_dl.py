import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torchvision.datasets as datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
import torchvision.models as models
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.metrics import classification_report
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import argparse
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)
import time
parser = argparse.ArgumentParser(description = 'What the program does')
parser.add_argument('--model_name', type = str)
parser.add_argument('--experiment_tag', type = str)
parser.add_argument('--project_dir', type = str)
args = parser.parse_args()

    
def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        normalize,
    ])

    test  = datasets.ImageFolder(test_path, transform= test_transform)
    test_dataloader = DataLoader(test, batch_size=bsz, shuffle = False)
    if model_name == 'alexnet':
        model = models.alexnet(pretrained = True)
        model.classifier[6] = nn.Linear(in_features = 4096, out_features = 2, bias = True)
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
    if model_name == 'resnet':
        model = models.resnet50(pretrained = True) # Pre-train on imagenet
        model.fc = nn.Linear(in_features = 2048, out_features = 2) # Convert to binary classification
    if model_name == 'googlenet':
        model = models.googlenet(pretrained = True)
        model.fc = nn.Linear(in_features = 1024, out_features = 2) 
    if model_name == 'inceptionv3':
        model = models.inception_v3(pretrained = True)
        model.fc = nn.Linear(in_features = 2048, out_features = 2)  
        model.aux_logits = False
        
    checkpoint = torch.load(output_model_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.cuda()

    def computeAUC(dataGT, dataPRED):
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        return roc_auc_score(datanpGT, datanpPRED)

    def class_report(dataGT, predCLASS):
        datanpGT = dataGT.cpu().numpy()
        datanppredCLASS = predCLASS.cpu().numpy()

        return classification_report(datanpGT, datanppredCLASS, digits = 3, output_dict = True)

    def model_test(model, dataloader):
        model.eval()
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        outCLASS = torch.FloatTensor().cuda()
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                image, labels = data
                image = image.float().cuda()
                labels = labels.float().cuda()
                output = model(image) # The output current is (N, C) where it is actually (N, 2)
                m = nn.Softmax(dim = 1)
                output_probabilities = m(output)
                class_pred = torch.argmax(output_probabilities, dim = 1)
                outPRED = torch.cat((outPRED, output_probabilities[:, -1]), 0)
                outGT = torch.cat((outGT, labels), 0)
                outCLASS = torch.cat((outCLASS, class_pred), 0)
        auc_test = computeAUC(outGT, outPRED)
        report = class_report(outGT, outCLASS)
       # print(report)
        return auc_test, report, outPRED, outCLASS, outGT
    auc_test,report, outPRED, outCLASS, outGT = model_test(model, test_dataloader)
   # print('TEST AUC:', auc_test)
    auc_scores.append(auc_test)
    accuracy_scores.append(report['accuracy'])
    ppv_scores.append(report['1.0']['precision'])
    npv_scores.append(report['0.0']['precision'])
    sensitivity_scores.append(report['1.0']['recall'])
    specificity_scores.append(report['0.0']['recall'])
    f1_scores.append(report['1.0']['f1-score'])
    
    parkinson_probability.append(outPRED.cpu().detach().numpy())
    class_predictions.append(outCLASS.cpu().detach().numpy())
    ground_truth.append(outGT.cpu().detach().numpy())
   # print(report['accuracy'], ',')


    
if __name__ == '__main__':
    args = parser.parse_args()
    ### You need to change the base data_folder based on the eperiment (Overall, Prevalent, Incident)
    # Your options are [alexnet, vgg, resnet, googlenet, inceptionv3]
    model_name = args.model_name
    base_folder = args.project_dir
    experiment_tag = args.experiment_tag
    bsz = 16
    
    auc_scores = [] 
    accuracy_scores = []
    ppv_scores = [] 
    npv_scores = [] 
    sensitivity_scores = [] 
    specificity_scores = [] 
    f1_scores = [] 
    parkinson_probability = [] 
    class_predictions = []
    ground_truth = [] 

    ########################

    if experiment_tag == 'overall': 
        data_extension = 'Overall_KFold_Data'
    if experiment_tag == 'prevalent':
        data_extension = 'Prevalent_KFold_Data'
    if experiment_tag == 'incident':
        data_extension = 'Incident_KFold_Data'      
        
        
        
    for j in range(1,6):
        for k in range(1,6):
            train_path = os.path.join(base_folder, 'data',  data_extension, 'R' + str(j),  str(k) , 'train')
            test_path =  os.path.join(base_folder, 'data', data_extension, 'R' + str(j), str(k) , 'test')
            output_model_dir = os.path.join(base_folder, 'models', experiment_tag, model_name,  model_name + 'weights' + '_' + str(j) + '_' + str(k) + '.pth')
            bsz = 64
            max_num_epochs = 100    
            main()



    output_base_dir_name = os.path.join(base_folder, 'results', experiment_tag, model_name)
    os.makedirs(output_base_dir_name, exist_ok = True)

    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'auc_scores.npy'), auc_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'accuracy_scores.npy'), accuracy_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'ppv_scores.npy'), ppv_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'npv_scores.npy'), npv_scores)
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
    

