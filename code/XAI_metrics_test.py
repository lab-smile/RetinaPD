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
from captum.attr import GuidedBackprop
from captum.attr import Saliency
from captum.metrics import sensitivity_max
from captum.metrics import infidelity
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz
from captum.attr import LayerActivation
from captum.attr import NoiseTunnel
import argparse
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)
import time
import matplotlib.font_manager
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

    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method that dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path

    
   # dataset_paths_var = ImageFolderWithPaths(test_path) # our custom dataset
   # paths_dataloader = DataLoader(dataset_paths_var)
    
    test = ImageFolderWithPaths(test_path, transform = test_transform)
    
    #test  = datasets.ImageFolder(test_path, transform= test_transform)
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
    model.cuda()
    
    def computeAUC(dataGT, dataPRED):
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        return roc_auc_score(datanpGT, datanpPRED)
    
    def perturb_fn(inputs):
        noise = torch.tensor(np.random.normal(0, 0.01, inputs.shape)).float().cuda()
        return (noise).cuda(), (inputs - noise).cuda()

    def model_test(model, dataloader):
        model.eval()
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        outCLASS = torch.FloatTensor().cuda()
        outATTRIBUTION = torch.FloatTensor().cuda()
        outINFIDELITY = torch.FloatTensor().cuda()
        outSENSITIVITY = torch.FloatTensor().cuda()
        attribution_method = attribution_method = GuidedBackprop(model)
       # attribution_method = Saliency(model)
        image_volume = torch.FloatTensor().cuda()
        paths_array = [] 
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                image, labels, paths = data
                image = image.float().cuda()
                image_volume = torch.cat((image_volume, image))
                attribution = attribution_method.attribute(image, target = 1)
                labels = labels.float().cuda()
                output = model(image) # The output current is (N, C) where it is actually (N, 2)
                m = nn.Softmax(dim = 1)
                output_probabilities = m(output)
                class_pred = torch.argmax(output_probabilities, dim = 1)
                # I want to convert
                outPRED = torch.cat((outPRED, output_probabilities), 0)
                outGT = torch.cat((outGT, labels), 0)
                outCLASS = torch.cat((outCLASS, class_pred), 0)
                outATTRIBUTION = torch.cat((outATTRIBUTION, attribution))
                sens = sensitivity_max(attribution_method.attribute, image, target = 1, n_perturb_samples = 32)
                infid = infidelity(model, perturb_fn, image, attribution, max_examples_per_batch = image.shape[0], n_perturb_samples = 32)
                paths_array.append(paths)
                outSENSITIVITY = torch.cat((outSENSITIVITY, sens))
                outINFIDELITY = torch.cat((outINFIDELITY, infid))

        return outPRED, outCLASS, outGT, outATTRIBUTION, image_volume, paths_array[0], outINFIDELITY, outSENSITIVITY


    outPRED, outCLASS, outGT, outATTRIBUTION, image_volume, paths_array, outINFIDELITY, outSENSITIVITY = model_test(model, test_dataloader)
    outINFIDELITY_scores.append(np.average(outINFIDELITY.cpu().detach().numpy()))
    outSENSITIVITY_scores.append(np.average(outSENSITIVITY.cpu().detach().numpy()))
    
    print('ITERATION: ', counter + 1, 'AVERAGE INFIDELITY', '{:.6f}'.format(np.average(outINFIDELITY.cpu().detach().numpy())))
    print('ITERATION: ', counter + 1, 'AVERAGE SENSITIVITY', '{:.3f}'.format(np.average(outSENSITIVITY.cpu().detach().numpy())))


    
if __name__ == '__main__':
    args = parser.parse_args()
    ### You need to change the base data_folder based on the eperiment (Overall, Prevalent, Incident)
    # Your options are [alexnet, vgg, resnet, googlenet, inceptionv3]

    model_name = args.model_name
    base_folder = args.project_dir
    experiment_tag = args.experiment_tag

    outINFIDELITY_scores = []
    outSENSITIVITY_scores = [] 
    counter = 0 
    
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
            bsz = 16 
            main()
            counter += 1
        
    output_base_dir_name = os.path.join(base_folder, 'results', experiment_tag, model_name)
    os.makedirs(output_base_dir_name, exist_ok = True)
   # model_name = 'alexnet'
              
    print('AVERAGE INFIDELITY', '{:.6f}'.format(np.average(outINFIDELITY_scores)*100))
    print('AVERAGE SENSITIVITY', '{:.3f}'.format(np.average(outSENSITIVITY_scores)))
    
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'XAIinfidelity.npy'), outINFIDELITY_scores)
    np.save(os.path.join(output_base_dir_name, model_name + '_' + 'XAIsensitivity.npy'),  outSENSITIVITY_scores)    
    
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'XAIinfidelity.npy'))
    print('SAVING:', os.path.join(output_base_dir_name, model_name + '_' + 'XAIsensitivity.npy'))
    

