import seaborn as sns
import matplotlib.pyplot as plt 
import argparse
import os
import numpy as np 
import pandas as pd 
from sklearn import metrics
parser = argparse.ArgumentParser(description = 'What the program does')
parser.add_argument('--experiment_tag', type = str)
parser.add_argument('--project_dir', type = str)
args = parser.parse_args()


def main():
    base_folder = os.path.join(args.project_dir,'results')
    experiment_tag = args.experiment_tag
                         
    ########## AUC ###############
    logistic_regression_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'logistic_regression',
                                               'logistic_regression_pd_probability.npy'), allow_pickle = True).tolist()
    elastic_net_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'elastic_net',
                                               'elastic_net_pd_probability.npy'), allow_pickle = True).tolist()
    svm_linear_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'svm_linear',
                                               'svm_linear_pd_probability.npy'), allow_pickle = True).tolist()
    svm_rbf_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'svm_rbf',
                                               'svm_rbf_pd_probability.npy'), allow_pickle = True).tolist()
    alexnet_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'alexnet',
                                               'alexnet_pd_probability.npy'), allow_pickle = True).tolist()


    vgg_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'vgg',
                                               'vgg_pd_probability.npy'), allow_pickle = True).tolist()

    googlenet_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'googlenet',
                                               'googlenet_pd_probability.npy'), allow_pickle = True).tolist()

    inceptionv3_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'inceptionv3',
                                               'inceptionv3_pd_probability.npy'), allow_pickle = True).tolist()

    resnet_prob = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'resnet',
                                               'resnet_pd_probability.npy'), allow_pickle = True).tolist()

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################

    logistic_regression_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'logistic_regression',
                                               'logistic_regression_ground_truth.npy'), allow_pickle = True).tolist()
    elastic_net_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'elastic_net',
                                               'elastic_net_ground_truth.npy'), allow_pickle = True).tolist()
    svm_linear_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'svm_linear',
                                               'svm_linear_ground_truth.npy'), allow_pickle = True).tolist()
    svm_rbf_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'svm_rbf',
                                               'svm_rbf_ground_truth.npy'), allow_pickle = True).tolist()
    alexnet_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'alexnet',
                                               'alexnet_ground_truth.npy'), allow_pickle = True).tolist()


    vgg_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'vgg',
                                               'vgg_ground_truth.npy'), allow_pickle = True).tolist()

    googlenet_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'googlenet',
                                               'googlenet_ground_truth.npy'), allow_pickle = True).tolist()

    inceptionv3_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'inceptionv3',
                                               'inceptionv3_ground_truth.npy'), allow_pickle = True).tolist()

    resnet_GT = np.load(os.path.join(base_folder,
                                               experiment_tag,
                                               'resnet',
                                               'resnet_ground_truth.npy'), allow_pickle = True).tolist()


    def curve_generation(model_GT, model_pred):
        fpr_mean    = np.linspace(0, 1, 50)
        interp_tprs = []
        for i in range(25):
            fpr, tpr, _ = metrics.roc_curve(model_GT[i], model_pred[i])
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 1.96*np.std(interp_tprs, axis=0) / np.sqrt(25)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = np.clip(tpr_mean-tpr_std, 0, 1)

        return fpr_mean, tpr_mean, tpr_upper, tpr_lower


    logistic_regression_fpr_mean, logistic_regression_tpr_mean, logistic_regression_tpr_upper, logistic_regression_tpr_lower = curve_generation(logistic_regression_GT,        logistic_regression_prob)
    elastic_net_fpr_mean, elastic_net_tpr_mean, elastic_net_tpr_upper, elastic_net_tpr_lower = curve_generation(elastic_net_GT, elastic_net_prob)
    svm_linear_fpr_mean, svm_linear_tpr_mean, svm_linear_tpr_upper, svm_linear_tpr_lower = curve_generation(svm_linear_GT, svm_linear_prob)
    svm_rbf_fpr_mean, svm_rbf_tpr_mean, svm_rbf_tpr_upper, svm_rbf_tpr_lower = curve_generation(svm_rbf_GT, svm_rbf_prob)
    alexnet_fpr_mean, alexnet_tpr_mean, alexnet_tpr_upper, alexnet_tpr_lower = curve_generation(alexnet_GT, alexnet_prob)
    vgg_fpr_mean, vgg_tpr_mean, vgg_tpr_upper, vgg_tpr_lower = curve_generation(vgg_GT, vgg_prob)
    googlenet_fpr_mean, googlenet_tpr_mean,googlenet_tpr_upper, googlenet_tpr_lower = curve_generation(googlenet_GT, googlenet_prob)
    inceptionv3_fpr_mean, inceptionv3_tpr_mean,inceptionv3_tpr_upper, inceptionv3_tpr_lower = curve_generation(inceptionv3_GT, inceptionv3_prob)
    resnet_fpr_mean, resnet_tpr_mean, resnet_tpr_upper, resnet_tpr_lower = curve_generation(resnet_GT, resnet_prob)

    font = {'fontname':'Arial'}
    plt.plot(logistic_regression_fpr_mean, logistic_regression_tpr_mean, color = 'black', linewidth = 2.5)
    plt.plot(elastic_net_fpr_mean, elastic_net_tpr_mean, color = 'aqua', linewidth = 2.5)
    plt.plot(svm_linear_fpr_mean, svm_linear_tpr_mean, color = 'brown', linewidth = 2.5)
    plt.plot(svm_rbf_fpr_mean, svm_rbf_tpr_mean, color = 'lime', linewidth = 2.5)
    plt.plot(alexnet_fpr_mean, alexnet_tpr_mean, color = 'b', linewidth = 2.5)
    plt.plot(vgg_fpr_mean, vgg_tpr_mean, color = 'red', linewidth = 2.5)
    plt.plot(googlenet_fpr_mean, googlenet_tpr_mean, color = 'purple', linewidth = 2.5)
    plt.plot(inceptionv3_fpr_mean, inceptionv3_tpr_mean, color = 'g', linewidth = 2.5)
    plt.plot(resnet_fpr_mean, resnet_tpr_mean, color = 'orange', linewidth = 2.5)

    x_points = [0, 1]
    y_points = [0, 1]

    plt.legend(['Logistic Regression (0.70)', 'Elastic Net (0.71)', 'Linear SVM (0.70)', 'RBF SVM (0.76)', 'AlexNet (0.78)', 'VGG-16 (0.77)', 'GoogleNet (0.72)', 'Inception-V3 (0.66)', 'ResNet-50 (0.72)'], fontsize = 10)

    plt.plot(x_points, y_points, linestyle='dashed', color = 'gray', linewidth = 2.5)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate ', fontsize = 14, **font)
    plt.ylabel('True Positive Rate', fontsize = 14, **font)

    plt.tick_params(axis='both', which='major', length = 8, labelsize = 12, width = 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.project_dir, 'results', args.experiment_tag + "_AUC_plot.pdf"))
    plt.title(args.experiment_tag + " Parkinson's ROC Curve", fontsize = 14, **font)

        
if __name__ == '__main__':
    main()