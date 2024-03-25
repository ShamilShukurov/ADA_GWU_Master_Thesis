

import pandas as pd
import time
from datasets import *
from base_learning import *
from balanced_learn import *
from learning_algorithms import *
from balancer import *
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def calculate_roc_data(y_true, y_scores):
    """
    Calculate the FPR, TPR, and AUC for given true labels and predicted scores.
    
    Parameters:
    - y_true (np.array): True binary labels.
    - y_scores (np.array): Target scores, can either be probability estimates of the positive class.
    
    Returns:
    - dict: A dictionary containing FPR, TPR, and AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}



def plot_roc_(roc_data, title):
    """
    Plots ROC curves for multiple models.
    
    Parameters:
    - roc_data (dict): A dictionary containing the false positive rates, true positive rates, and AUC scores for each model.
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(6, 4))
    for model_name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'], label=f'{model_name} (AUC = {data["auc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def make_experiment(dfs:List[Experiment_Dataset], learning_algorithms:List[BaseLearningAlgorithm])->pd.DataFrame:
    """
    Trains and evaluates each learning algorithm on each dataset.
    
    Parameters:
    - dfs (list of Experiment_Dataset): List of datasets to be used in the experiments.
    - learning_algorithms (list of BaseLearningAlgorithm): List of learning algorithms to be applied.
    
    Returns:
    - pd.DataFrame: A concatenated DataFrame containing evaluation reports for all experiments.
    """
    start_time = time.time()
    all_reports = []  # List to store individual reports

    for dataset in dfs:
        roc_data_train = {}
        roc_data_test = {}

        for algorithm in learning_algorithms:
            print(f"Starting experiment with dataset '{dataset.name}' and model '{algorithm.name}'...")
            
            # Train and evaluate the model on the current dataset
            report = algorithm.train_eval(dataset.X_train, dataset.y_train,
                                          dataset.X_test, dataset.y_test)
            
            # Adding name of the datasets to the report for clarity
            report['Dataset'] = dataset.name
            # report['Model_Name'] = algorithm.name
            
            # Displaying the report for the current experiment
            print(report)
            
            # Appending the report to the list of all reports
            all_reports.append(report)

            probas_train = algorithm.predict_proba(dataset.X_train)[:, 1]  # Assuming binary classification
            probas_test = algorithm.predict_proba(dataset.X_test)[:, 1]
            
            roc_train = calculate_roc_data(dataset.y_train, probas_train)
            roc_test = calculate_roc_data(dataset.y_test, probas_test)
            
            roc_data_train[algorithm.name] = roc_train
            roc_data_test[algorithm.name] = roc_test
        plot_roc_(roc_data_train, title=f"ROC Curves for Training Data on {dataset.name}")
        plot_roc_(roc_data_test, title=f"ROC Curves for Test Data on {dataset.name}")

    # Return all reports in one DataFrame
    training_duration = time.time() - start_time
    print(f"Experiment Done in {training_duration} seconds")
    return pd.concat(all_reports, ignore_index=True)