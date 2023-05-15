import seaborn as sns
import matplotlib.pyplot as plt 
import argparse
import os
import numpy as np 
import pandas as pd 
from sklearn import metrics
parser = argparse.ArgumentParser(description = 'What the program does')
#parser.add_argument('--experiment_tag', type = str)
parser.add_argument('--project_dir', type = str)
args = parser.parse_args()


def main():
    base_folder = os.path.join(args.project_dir,'results')
    #experiment_tag = args.experiment_tag

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################

    alexnet_overall_infid = np.log10(np.load(os.path.join(base_folder,
                                               'overall',
                                               'alexnet',
                                               'alexnet_XAIinfidelity.npy'), allow_pickle = True)).tolist()


    vgg_overall_infid = np.log10(np.load(os.path.join(base_folder,
                                               'overall',
                                               'vgg',
                                               'vgg_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    googlenet_overall_infid = np.log10(np.load(os.path.join(base_folder,
                                               'overall',
                                               'googlenet',
                                               'googlenet_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    inceptionv3_overall_infid = np.log10(np.load(os.path.join(base_folder,
                                               'overall',
                                               'inceptionv3',
                                               'inceptionv3_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    resnet_overall_infid = np.log10(np.load(os.path.join(base_folder,
                                               'overall',
                                               'resnet',
                                               'resnet_XAIinfidelity.npy'), allow_pickle = True)).tolist()
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################

    alexnet_prev_infid = np.log10(np.load(os.path.join(base_folder,
                                               'prevalent',
                                               'alexnet',
                                               'alexnet_XAIinfidelity.npy'), allow_pickle = True)).tolist()


    vgg_prev_infid = np.log10(np.load(os.path.join(base_folder,
                                               'prevalent',
                                               'vgg',
                                               'vgg_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    googlenet_prev_infid = np.log10(np.load(os.path.join(base_folder,
                                               'prevalent',
                                               'googlenet',
                                               'googlenet_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    inceptionv3_prev_infid = np.log10(np.load(os.path.join(base_folder,
                                               'prevalent',
                                               'inceptionv3',
                                               'inceptionv3_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    resnet_prev_infid = np.log10(np.load(os.path.join(base_folder,
                                               'prevalent',
                                               'resnet',
                                               'resnet_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################

    alexnet_inc_infid = np.log10(np.load(os.path.join(base_folder,
                                               'incident',
                                               'alexnet',
                                               'alexnet_XAIinfidelity.npy'), allow_pickle = True)).tolist()


    vgg_inc_infid = np.log10(np.load(os.path.join(base_folder,
                                               'incident',
                                               'vgg',
                                               'vgg_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    googlenet_inc_infid = np.log10(np.load(os.path.join(base_folder,
                                               'incident',
                                               'googlenet',
                                               'googlenet_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    inceptionv3_inc_infid = np.log10(np.load(os.path.join(base_folder,
                                               'incident',
                                               'inceptionv3',
                                               'inceptionv3_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    resnet_inc_infid = np.log10(np.load(os.path.join(base_folder,
                                               'incident',
                                               'resnet',
                                               'resnet_XAIinfidelity.npy'), allow_pickle = True)).tolist()

    dataset = pd.DataFrame()
    dataset['Overall Alexnet'] = alexnet_overall_infid
    dataset['Prevalent AlexNet'] = alexnet_prev_infid
    dataset['Incident Alexnet'] = alexnet_inc_infid
    dataset['Overall VGG-16'] = vgg_overall_infid
    dataset['Prevalent VGG-16'] = vgg_prev_infid
    dataset['Incident VGG-16'] = vgg_inc_infid
    dataset['Overall GoogleNet'] = googlenet_overall_infid
    dataset['Prevalent GoogleNet'] = googlenet_prev_infid
    dataset['Incident GoogleNet'] = googlenet_inc_infid
    dataset['Overall Inception-V3'] = inceptionv3_overall_infid
    dataset['Prevalent Inception-V3'] = inceptionv3_prev_infid
    dataset['Incident Inception-V3'] = inceptionv3_inc_infid
    dataset['Overall ResNet-50'] = resnet_overall_infid
    dataset['Prevalent ResNet-50'] = resnet_prev_infid
    dataset['Incident ResNet-50'] = resnet_inc_infid


    ax = sns.boxplot(data=dataset, width = 0.5, palette = "YlOrBr", showfliers = False)
    for line in ax.get_lines():
        line.set_color('black')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize = 13)
    ax.set_ylabel('Infidelity (Log10)', fontsize = 14)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    ax.xaxis.set_tick_params(width=1, length = 8)
    ax.yaxis.set_tick_params(width=1, length = 8)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.project_dir, 'results',  "infidelity_plot.svg"), format = 'svg')
    plt.savefig(os.path.join(args.project_dir, 'results',  "infidelity_plot.pdf"), format = 'pdf')

        
if __name__ == '__main__':
    main()