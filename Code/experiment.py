

import pandas as pd
import time
from datasets import *
from base_learning import *
from balanced_learn import *
from learning_algorithms import *
from balancer import *
from typing import List

def make_experiment(dfs:List[Experiment_Dataset], learning_algorithms:List[BaseLearningAlgorithm])->pd.DataFrame:
    """
    Trains and evaluates each learning algorithm on each dataset.
    
    Parameters:
    - dfs (list of Experiment_Dataset): List of datasets to be used in the experiments.
    - learning_algorithms (list of BaseLearningAlgorithm): List of learning algorithms to be applied.
    
    Returns:
    - pd.DataFrame: A concatenated DataFrame containing evaluation reports for all experiments.
    """
    all_reports = []  # List to store individual reports

    for dataset in dfs:
        for algorithm in learning_algorithms:
            print(f"Starting experiment with dataset '{dataset.name}' and model '{algorithm.name}'...")
            
            # Train and evaluate the model on the current dataset
            report = algorithm.train_eval(dataset.X_train, dataset.y_train,
                                          dataset.X_test, dataset.y_test)
            
            # Adding extra columns to the report for clarity
            report['Dataset_Name'] = dataset.name
            report['Model_Name'] = algorithm.name
            
            # Displaying the report for the current experiment
            print(report)
            
            # Appending the report to the list of all reports
            all_reports.append(report)

    # Return all reports in one DataFrame
    return pd.concat(all_reports, ignore_index=True)