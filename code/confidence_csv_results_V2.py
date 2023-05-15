import numpy as np
import scipy.stats as st
import csv
import argparse
import os 

parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
parser.add_argument('--project_dir', type = str)           # positional argument
parser.add_argument('--model_name', type = str)           # positional argument
parser.add_argument('--experiment_tag', type = str)           # positional argument
def main():
# Edit paths in `np.load()` to ensure that the right results are being accessed
    args = parser.parse_args()

    with open(os.path.join(args.project_dir, 'results', 'results_V2.csv'), 'a') as file:
        writer = csv.writer(file, delimiter=',')

        #### Edit this for each experiment, i.e. "Overall, Alexnet" -> "Prevalent, AlexNet"
        #writer.writerow([args.experiment_tag + ',' + args.model_name, "Metrics", "Average (95% Confidence Interval)"])
        #writer.writerow([args.experiment_tag, args.model_name, "AUC", "Accuracy", "PPV", "NPV", "Sensitivity", "Specificity", "F1"])
        #################################

        auc = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_auc_scores.npy'))
        auc_interval = st.t.interval(0.95, len(auc)-1, loc=np.mean(auc), scale=st.sem(auc))
        auc_mean = str(np.round(np.mean(auc), 2))
        auc_lower = str(np.round(auc_interval[0],2))
        auc_upper = str(np.round(auc_interval[1], 2))
       # print("AUC", np.mean(X) + '(' + lower_A + ',' + upper_A + ')')

        acc = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_accuracy_scores.npy'))
        acc_interval = st.t.interval(0.95, len(acc)-1, loc=np.mean(acc), scale=st.sem(acc))
        acc_mean = str(np.round(np.mean(acc), 2))
        acc_lower = str(np.round(acc_interval[0],2))
        acc_upper = str(np.round(acc_interval[1], 2))
      #  print("Accuracy", np.mean(X), '(' + lower_A + ',' + 'upper A' + ')')


        ppv = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_ppv_scores.npy'))
        ppv_interval = st.t.interval(0.95, len(ppv)-1, loc=np.mean(ppv), scale=st.sem(ppv))
        ppv_mean = str(np.round(np.mean(ppv), 2))
        ppv_lower = str(np.round(ppv_interval[0],2))
        ppv_upper = str(np.round(ppv_interval[1], 2))


        npv = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_npv_scores.npy'))
        npv_interval = st.t.interval(0.95, len(npv)-1, loc=np.mean(npv), scale=st.sem(npv))
        npv_mean = str(np.round(np.mean(npv), 2))
        npv_lower = str(np.round(npv_interval[0],2))
        npv_upper = str(np.round(npv_interval[1], 2))


        sen = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_sensitivity_scores.npy'))
        sen_interval = st.t.interval(0.95, len(sen)-1, loc=np.mean(sen), scale=st.sem(sen))
        sen_mean = str(np.round(np.mean(sen), 2))
        sen_lower = str(np.round(sen_interval[0],2))
        sen_upper = str(np.round(sen_interval[1], 2))


        spe = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_specificity_scores.npy'))
        spe_interval = st.t.interval(0.95, len(spe)-1, loc=np.mean(spe), scale=st.sem(spe))
        spe_mean = str(np.round(np.mean(spe), 2))
        spe_lower = str(np.round(spe_interval[0],2))
        spe_upper = str(np.round(spe_interval[1], 2))


        f1 = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_f1_scores.npy'))
        f1_interval = st.t.interval(0.95, len(f1)-1, loc=np.mean(f1), scale=st.sem(f1))
        f1_mean = str(np.round(np.mean(f1), 2))
        f1_lower = str(np.round(f1_interval[0],2))
        f1_upper = str(np.round(f1_interval[1], 2))

        writer.writerow([args.model_name, 
                         auc_mean + ' (' + auc_lower + ', ' + auc_upper + ')',
                         acc_mean + ' (' + acc_lower + ', ' + acc_upper + ')',
                         ppv_mean + ' (' + ppv_lower + ', ' + ppv_upper + ')',
                         npv_mean + ' (' + npv_lower + ', ' + npv_upper + ')',
                         sen_mean + ' (' + sen_lower + ', ' + sen_upper + ')',
                         spe_mean + ' (' + spe_lower + ', ' + spe_upper + ')',
                         f1_mean + ' (' + f1_lower + ', ' + f1_upper + ')',
                        ])

if __name__ == '__main__':
    main()