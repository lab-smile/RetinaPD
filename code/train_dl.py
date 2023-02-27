import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

import torchvision.models as models
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.models as models
import numpy as np
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

#import tensorboard_logger as tb_logger
#from torchsampler import ImbalancedDatasetSampler
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse 

parser = argparse.ArgumentParser(description = 'What the program does')
parser.add_argument('--model_name', type = str)
parser.add_argument('--experiment_tag', type = str)
parser.add_argument('--project_dir', type = str)
parser.add_argument('--epochs', type =int)


def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p =0.4),
        transforms.RandomVerticalFlip(p = 0.4),
        transforms.RandomRotation(50),
        transforms.ToTensor(),
        transforms.Resize(256),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        normalize,
    ])
 
    train = datasets.ImageFolder(train_path, transform= train_transform)
    val  = datasets.ImageFolder(test_path, transform= train_transform)

    train_dataloader = DataLoader(train, batch_size=train_bsz, shuffle = True)
    val_dataloader = DataLoader(val, batch_size = val_bsz, shuffle = True)    
    
    
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
        
 
        
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    def model_train(model, dataloader):
        model.train()
        tr_loss = []
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            image, labels = data
            image = image.float().cuda()
            labels = labels.long().cuda()
            output = model(image)
       
            #class_pred = torch.argmax(output, dim = 1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())


       # print('Training Loss:', np.average(tr_loss, axis=0))
        avg_tr_loss = np.average(tr_loss, axis=0)
        return avg_tr_loss

    def model_eval(model, dataloader):
        model.eval()
        val_loss = []
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                optimizer.zero_grad()
                image, labels = data
                image = image.float().cuda()
                labels = labels.long().cuda()
                output = model(image)
                loss = criterion(output, labels)
                val_loss.append(loss.item())
      #  print('Validation Loss:', np.average(val_loss, axis=0))
        avg_val_loss = np.average(val_loss, axis=0)
        return avg_val_loss

    # initialize val loss counter to be high for early stopping method
    val_loss_counter = [100000000] # Initialize to a very large number

    for epoch in range(max_num_epochs):
        training_loss = model_train(model, train_dataloader)
        validation_loss = model_eval(model, val_dataloader)
       # print(epoch,'Training Loss:', training_loss, 'Validation Loss', validation_loss)
        if validation_loss <= np.amin(val_loss_counter):
            torch.save({'model_state_dict': model.state_dict(),
            }, output_model_dir )
            #torch.save({'model_state_dict': model.state_dict(), output_model_dir})            
            print('Model has been saved to', output_model_dir)
 
        val_loss_counter.append(validation_loss)
    
if __name__ == '__main__':
    args = parser.parse_args()
    ### You need to change the base data_folder based on the eperiment (Overall, Prevalent, Incident)
    # Your options are [alexnet, vgg, resnet, googlenet, inceptionv3]
    model_name = args.model_name
    base_folder = args.project_dir
    experiment_tag = args.experiment_tag

    
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
            os.makedirs(os.path.join(base_folder, 'models', experiment_tag, model_name), exist_ok = True)
            learning_rate = 1e-4
            train_bsz = 64
            val_bsz = 64
            max_num_epochs = args.epochs
            main()
            
    print('Confirm Model Output Name:', model_name)
    

