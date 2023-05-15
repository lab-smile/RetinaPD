import seaborn as sns
import matplotlib.pyplot as plt 
import argparse
import os
import numpy as np 
import pandas as pd 
from sklearn import metrics
import matplotlib.font_manager
parser = argparse.ArgumentParser(description = 'What the program does')
parser.add_argument('--experiment_tag', type = str)
parser.add_argument('--project_dir', type = str)
args = parser.parse_args()


def main():
    base_folder = os.path.join(args.project_dir,'results')
    experiment_tag = args.experiment_tag


    font = {'fontname':'Arial'}

    logistic_regression = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'logistic_regression',
                                               'logistic_regression_accuracy_scores.npy'), allow_pickle = True).tolist()
    elastic_net = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'elastic_net',
                                               'elastic_net_accuracy_scores.npy'), allow_pickle = True).tolist()
    svm_linear = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'svm_linear',
                                               'svm_linear_accuracy_scores.npy'), allow_pickle = True).tolist()
    svm_rbf = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'svm_rbf',
                                               'svm_rbf_accuracy_scores.npy'), allow_pickle = True).tolist()
    alexnet = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'alexnet',
                                               'alexnet_accuracy_scores.npy'), allow_pickle = True).tolist()


    vgg = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'vgg',
                                               'vgg_accuracy_scores.npy'), allow_pickle = True).tolist()

    googlenet = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'googlenet',
                                               'googlenet_accuracy_scores.npy'), allow_pickle = True).tolist()

    inceptionv3 = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'inceptionv3',
                                               'inceptionv3_accuracy_scores.npy'), allow_pickle = True).tolist()

    resnet = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'resnet',
                                               'resnet_accuracy_scores.npy'), allow_pickle = True).tolist()



    model_names = ['Logistic Regression', 'Elastic Net', 'Linear SVM', 'RBF SVM', 'AlexNet', 'VGG-16', 'GoogleNet', 'InceptionV3', 'ResNet-50']
    scores = [logistic_regression, elastic_net, svm_linear, svm_rbf, alexnet, vgg, googlenet, inceptionv3, resnet]

    dataset = pd.DataFrame()
    dataset['Logistic Regression'] = scores[0]
    dataset['Elastic Net'] = scores[1]
    dataset['Linear SVM'] = scores[2]
    dataset['RBF SVM'] = scores[3]
    dataset['AlexNet'] = scores[4]
    dataset['VGG-16'] = scores[5]
    dataset['GoogleNet'] = scores[6]
    dataset['Inception-V3'] = scores[7]
    dataset['ResNet-50'] = scores[8]
    
    if args.experiment_tag == 'overall':
        palette_color = 'BuGn'
    if args.experiment_tag == 'prevalent':
        palette_color = 'PuBu'
    if args.experiment_tag == 'incident':
        palette_color = 'RdPu'
    #plt.figure(figsize = (11,14))
    ax = sns.boxplot(data=dataset, width = 0.4, palette = palette_color)
    #sns.set_style("whitegrid")
    for line in ax.get_lines():
        line.set_color('black')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize = 13, **font)
    ax.set_ylabel('Accuracy', fontsize = 14, **font)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    ax.xaxis.set_tick_params(width=1, length = 8)
    ax.yaxis.set_tick_params(width=1, length = 8)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    if args.experiment_tag == 'overall':
        plt.title("Overall" + " Parkinson's Classification", fontsize = 14, **font)
    if args.experiment_tag == 'prevalent':
        plt.title("Prevalent" + " Parkinson's Classification", fontsize = 14, **font)
    if args.experiment_tag == 'incident':
        plt.title("Incident" + " Parkinson's Classification", fontsize = 14, **font)
    
    plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) 
    plt.tight_layout()
    plt.savefig(os.path.join(args.project_dir, 'results', args.experiment_tag + "_accuracy_plot.svg"), format = 'svg')
    plt.savefig(os.path.join(args.project_dir, 'results', args.experiment_tag + "_accuracy_plot.pdf"), format = 'pdf')
    plt.show()
                         
                               
                               
        
if __name__ == '__main__':
    main()