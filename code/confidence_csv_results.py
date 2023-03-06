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

    with open(os.path.join(args.project_dir, 'results', 'results.csv'), 'a') as file:
        writer = csv.writer(file, delimiter=',')

        #### Edit this for each experiment, i.e. "Overall, Alexnet" -> "Prevalent, AlexNet"
        writer.writerow([args.experiment_tag + ',' + args.model_name, "Metrics", "Average (95% Confidence Interval)"])
        model = args.model_name
        experiment = args.experiment_tag
        #################################

        X = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_auc_scores.npy'))
        A = st.t.interval(0.95, len(X)-1, loc=np.mean(X), scale=st.sem(X))
        mean_A = str(np.round(np.mean(X), 2))
        lower_A = str(np.round(A[0],2))
        upper_A = str(np.round(A[1], 2))
       # print("AUC", np.mean(X) + '(' + lower_A + ',' + upper_A + ')')
        writer.writerow(["", "AUC", mean_A + '(' + lower_A + ',' + upper_A + ')'])

        X = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_accuracy_scores.npy'))
        A = st.t.interval(0.95, len(X)-1, loc=np.mean(X), scale=st.sem(X))
        mean_A = str(np.round(np.mean(X), 2))
        lower_A = str(np.round(A[0],2))
        upper_A = str(np.round(A[1], 2))
      #  print("Accuracy", np.mean(X), '(' + lower_A + ',' + 'upper A' + ')')
        writer.writerow(["", "Accuracy", mean_A + '(' + lower_A + ',' + upper_A + ')'])

        X = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_ppv_scores.npy'))
        A = st.t.interval(0.95, len(X)-1, loc=np.mean(X), scale=st.sem(X))
        mean_A = str(np.round(np.mean(X), 2))
        lower_A = str(np.round(A[0],2))
        upper_A = str(np.round(A[1], 2))
      #  print("PPV", np.mean(X), '(' + lower_A + ',' + 'upper A' + ')')
        writer.writerow(["", "PPV", mean_A + '(' + lower_A + ',' + upper_A + ')'])

        X = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_npv_scores.npy'))
        A = st.t.interval(0.95, len(X)-1, loc=np.mean(X), scale=st.sem(X))
        mean_A = str(np.round(np.mean(X), 2))
        lower_A = str(np.round(A[0],2))
        upper_A = str(np.round(A[1], 2))
      #  print("NPV", np.mean(X), '(' + lower_A + ',' + 'upper A' + ')')
        writer.writerow(["", "NPV", mean_A + '(' + lower_A + ',' + upper_A + ')'])

        X = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_sensitivity_scores.npy'))
        A = st.t.interval(0.95, len(X)-1, loc=np.mean(X), scale=st.sem(X))
        mean_A = str(np.round(np.mean(X), 2))
        lower_A = str(np.round(A[0],2))
        upper_A = str(np.round(A[1], 2))
    #    print("Sensitivity", np.mean(X), '(' + lower_A + ',' + 'upper A' + ')')
        writer.writerow(["", "Sensitivity", mean_A + '(' + lower_A + ',' + upper_A + ')'])

        X = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_specificity_scores.npy'))
        A = st.t.interval(0.95, len(X)-1, loc=np.mean(X), scale=st.sem(X))
        mean_A = str(np.round(np.mean(X), 2))
        lower_A = str(np.round(A[0],2))
        upper_A = str(np.round(A[1], 2))
     #   print("Specificity", np.mean(X), '(' + lower_A + ',' + 'upper A' + ')')
        writer.writerow(["", "Specificity", mean_A + '(' + lower_A + ',' + upper_A + ')'])

        X = np.load(os.path.join(args.project_dir, 'results',  args.experiment_tag, args.model_name, args.model_name + '_f1_scores.npy'))
        A = st.t.interval(0.95, len(X)-1, loc=np.mean(X), scale=st.sem(X))
        mean_A = str(np.round(np.mean(X), 2))
        lower_A = str(np.round(A[0],2))
        upper_A = str(np.round(A[1], 2))
      #  print("F1-Score", np.mean(X), '(' + lower_A + ',' + 'upper A' + ')')
        writer.writerow(["", "F1-Score", mean_A + '(' + lower_A + ',' + upper_A + ')'])

if __name__ == '__main__':
    main()