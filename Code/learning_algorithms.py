import numpy as np
import pandas as pd
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from base_learning import BaseLearningAlgorithm
from os.path import join
from learning_algorithms import *
from utils import *


class SVMClassifier(BaseLearningAlgorithm):
    """SVM Classifier implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name = 'SimpleSVC', kernel='rbf', C=5.0, class_weight=None, verbose=False):
        self.model = SVC(kernel=kernel, C=C, probability=True, class_weight=class_weight, verbose=verbose)
        self.alg_name = alg_name
        self.kernel = kernel
        self.C = C
        self.class_weight = class_weight
        self.verbose = verbose
        
    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Fit the SVM model to the training data."""
        self.model.fit(x_train, y_train)
    
    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predict using the fitted SVM model."""
        return self.model.predict(x_test)
    
    def train_eval(self, x_train: pd.DataFrame, y_train: np.array, x_test: pd.DataFrame, 
                   y_test: np.array, x_val: pd.DataFrame = None, y_val: np.array = None, 
                   save_model: bool = True) -> pd.DataFrame:
        """Train and evaluate the SVM model on both training and test datasets."""
        self.fit(x_train, y_train, x_val, y_val)

        # Predictions and probabilities for test data
        predictions_test = self.predict(x_test)
        probabilities_test = self.model.predict_proba(x_test)[:, 1]

        # Predictions and probabilities for train data
        predictions_train = self.predict(x_train)
        probabilities_train = self.model.predict_proba(x_train)[:, 1]

        # Evaluation metrics for test data
        f1_test = f1_score(y_test, predictions_test, zero_division=0)
        accuracy_test = accuracy_score(y_test, predictions_test)
        precision_test = precision_score(y_test, predictions_test, zero_division=0)
        recall_test = recall_score(y_test, predictions_test, zero_division=0)
        auc_score_test = roc_auc_score(y_test, probabilities_test)

        # Evaluation metrics for train data
        f1_train = f1_score(y_train, predictions_train, zero_division=0)
        accuracy_train = accuracy_score(y_train, predictions_train)
        precision_train = precision_score(y_train, predictions_train, zero_division=0)
        recall_train = recall_score(y_train, predictions_train, zero_division=0)
        auc_score_train = roc_auc_score(y_train, probabilities_train)

        # Plot ROC curve for both test and train data
        self.plot_roc_curve(y_test, probabilities_test, 'Test')
        self.plot_roc_curve(y_train, probabilities_train, 'Train')

        evaluation_report = pd.DataFrame({
            'Model': [self.alg_name, self.alg_name],
            'Dataset': ['Train', 'Test'],
            'Accuracy': [accuracy_train, accuracy_test],
            'F1 Score': [f1_train, f1_test],
            'Precision': [precision_train, precision_test],
            'Recall': [recall_train, recall_test],
            'AUC Score': [auc_score_train, auc_score_test]
        })

        if save_model:
            # Implement model saving logic here
            pass

        return evaluation_report
    
    # def plot_roc_curve(self, y_true: np.array, probabilities: np.array, dataset_label: str) -> None:
    #     """Plot the ROC curve for a given dataset."""
    #     fpr, tpr, _ = roc_curve(y_true, probabilities)
    #     roc_auc = auc(fpr, tpr)

    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve ({dataset_label} - area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'Receiver Operating Characteristic - {dataset_label}')
    #     plt.legend(loc="lower right")
    #     plt.show()


    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return f"{self.alg_name}_{self.kernel}_C{self.C}"
    


class XGBoostClassifier(BaseLearningAlgorithm):
    """XGBoost Classifier implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name='SimpleXGB', max_depth=3, learning_rate=0.1, 
                 n_estimators=100, verbosity=0, objective='binary:logistic', 
                 booster='gbtree', class_weight=1):
        self.model = xgb.XGBClassifier(max_depth=max_depth, 
                                       learning_rate=learning_rate, 
                                       n_estimators=n_estimators, 
                                       verbosity=verbosity, 
                                       objective=objective, 
                                       booster=booster, 
                                       use_label_encoder=False, 
                                       scale_pos_weight=class_weight)
        self.alg_name = alg_name
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.objective = objective
        self.booster = booster
        self.class_weight = class_weight

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Fit the XGBoost model to the training data."""
        self.model.fit(x_train, y_train, eval_metric='logloss')

    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predict using the fitted XGBoost model."""
        return self.model.predict(x_test)

    def train_eval(self, x_train: pd.DataFrame, y_train: np.array, x_test: pd.DataFrame, y_test: np.array, x_val: pd.DataFrame = None, y_val: np.array = None, save_model: bool = True) -> pd.DataFrame:
        """Train and evaluate the XGBoost model on both training and test datasets."""
        self.fit(x_train, y_train, x_val, y_val)

        predictions_test = self.predict(x_test)
        probabilities_test = self.model.predict_proba(x_test)[:, 1]

        predictions_train = self.predict(x_train)
        probabilities_train = self.model.predict_proba(x_train)[:, 1]

        # Evaluation metrics for test data
        f1_test = f1_score(y_test, predictions_test, zero_division=0)
        accuracy_test = accuracy_score(y_test, predictions_test)
        precision_test = precision_score(y_test, predictions_test, zero_division=0)
        recall_test = recall_score(y_test, predictions_test, zero_division=0)
        auc_score_test = roc_auc_score(y_test, probabilities_test)

        # Evaluation metrics for train data
        f1_train = f1_score(y_train, predictions_train, zero_division=0)
        accuracy_train = accuracy_score(y_train, predictions_train)
        precision_train = precision_score(y_train, predictions_train, zero_division=0)
        recall_train = recall_score(y_train, predictions_train, zero_division=0)
        auc_score_train = roc_auc_score(y_train, probabilities_train)

        # Plot ROC curve for both test and train data
        self.plot_roc_curve(y_test, probabilities_test, 'Test')
        self.plot_roc_curve(y_train, probabilities_train, 'Train')

        evaluation_report = pd.DataFrame({
            'Model': [self.alg_name, self.alg_name],
            'Dataset': ['Train', 'Test'],
            'Accuracy': [accuracy_train, accuracy_test],
            'F1 Score': [f1_train, f1_test],
            'Precision': [precision_train, precision_test],
            'Recall': [recall_train, recall_test],
            'AUC Score': [auc_score_train, auc_score_test]
        })

        return evaluation_report

    # def plot_roc_curve(self, y_true: np.array, probabilities: np.array, dataset_label: str) -> None:
    #     """Plot the ROC curve for a given dataset."""
    #     fpr, tpr, _ = roc_curve(y_true, probabilities)
    #     roc_auc = auc(fpr, tpr)

    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve ({dataset_label} - area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'Receiver Operating Characteristic - {dataset_label}')
    #     plt.legend(loc="lower right")
    #     plt.show()

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return f"{self.alg_name}_{self.booster}_depth{self.max_depth}_lr{self.learning_rate}_est{self.n_estimators}"
    

class LogisticRegressionClassifier(BaseLearningAlgorithm):
    """Logistic Regression implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name='SimpleLogisticRegression', 
                 penalty='l2', C=1.0, class_weight=None, solver='lbfgs', max_iter=100, verbose=0):
        self.model = LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, 
                                        solver=solver, max_iter=max_iter, verbose=verbose)
        self.alg_name = alg_name
        self.penalty = penalty
        self.C = C
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Fit the Logistic Regression model to the training data."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predict using the fitted Logistic Regression model."""
        return self.model.predict(x_test)

    def train_eval(self, x_train: pd.DataFrame, y_train: np.array, 
                   x_test: pd.DataFrame, y_test: np.array, x_val: pd.DataFrame = None, 
                   y_val: np.array = None, save_model: bool = True) -> pd.DataFrame:
        """Train and evaluate the Logistic Regression model."""
        self.fit(x_train, y_train, x_val, y_val)

        predictions_test = self.predict(x_test)
        probabilities_test = self.model.predict_proba(x_test)[:, 1]

        predictions_train = self.predict(x_train)
        probabilities_train = self.model.predict_proba(x_train)[:, 1]

        # Evaluation metrics for test data
        f1_test = f1_score(y_test, predictions_test, zero_division=0)
        accuracy_test = accuracy_score(y_test, predictions_test)
        precision_test = precision_score(y_test, predictions_test, zero_division=0)
        recall_test = recall_score(y_test, predictions_test, zero_division=0)
        auc_score_test = roc_auc_score(y_test, probabilities_test)

        # Evaluation metrics for train data
        f1_train = f1_score(y_train, predictions_train, zero_division=0)
        accuracy_train = accuracy_score(y_train, predictions_train)
        precision_train = precision_score(y_train, predictions_train, zero_division=0)
        recall_train = recall_score(y_train, predictions_train, zero_division=0)
        auc_score_train = roc_auc_score(y_train, probabilities_train)

        # Plot ROC curve for both test and train data
        self.plot_roc_curve(y_test, probabilities_test, 'Test')
        self.plot_roc_curve(y_train, probabilities_train, 'Train')

        evaluation_report = pd.DataFrame({
            'Model': [self.alg_name, self.alg_name],
            'Dataset': ['Train', 'Test'],
            'Accuracy': [accuracy_train, accuracy_test],
            'F1 Score': [f1_train, f1_test],
            'Precision': [precision_train, precision_test],
            'Recall': [recall_train, recall_test],
            'AUC Score': [auc_score_train, auc_score_test]
        })

        if save_model:
            # Implement model saving logic here, possibly using joblib or pickle
            pass


        return evaluation_report

    # def plot_roc_curve(self, y_true: np.array, probabilities: np.array, dataset_label: str) -> None:
    #     """Plot the ROC curve for a given dataset."""
    #     fpr, tpr, _ = roc_curve(y_true, probabilities)
    #     roc_auc = auc(fpr, tpr)

    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve ({dataset_label} - area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'Receiver Operating Characteristic - {dataset_label}')
    #     plt.legend(loc="lower right")
    #     plt.show()

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return f"{self.alg_name}_{self.penalty}_C{self.C}"


class EasyEnsemble(BaseLearningAlgorithm):
    """Easy Ensemble Classifier implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, n_estimators=10, random_state=42, n_jobs=-1):
        self.model = EasyEnsembleClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
        self.alg_name = "EasyEnsemble"
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Fit the Easy Ensemble model to the training data."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predict using the fitted Easy Ensemble model."""
        return self.model.predict(x_test)

    def train_eval(self, x_train: pd.DataFrame, y_train: np.array, 
                   x_test: pd.DataFrame, y_test: np.array, 
                   x_val: pd.DataFrame = None, y_val: np.array = None, save_model: bool = True) -> pd.DataFrame:
        """Train and evaluate the Easy Ensemble model."""
        self.fit(x_train, y_train, x_val, y_val)

        predictions_test = self.predict(x_test)
        probabilities_test = self.model.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class

        predictions_train = self.predict(x_train)
        probabilities_train = self.model.predict_proba(x_train)[:, 1]  # Get probabilities for the positive class

        # Evaluation metrics for test data
        f1_test = f1_score(y_test, predictions_test, zero_division=0)
        accuracy_test = accuracy_score(y_test, predictions_test)
        precision_test = precision_score(y_test, predictions_test, zero_division=0)
        recall_test = recall_score(y_test, predictions_test, zero_division=0)
        auc_score_test = roc_auc_score(y_test, probabilities_test)

        # Evaluation metrics for train data
        f1_train = f1_score(y_train, predictions_train, zero_division=0)
        accuracy_train = accuracy_score(y_train, predictions_train)
        precision_train = precision_score(y_train, predictions_train, zero_division=0)
        recall_train = recall_score(y_train, predictions_train, zero_division=0)
        auc_score_train = roc_auc_score(y_train, probabilities_train)


        evaluation_report = pd.DataFrame({
            'Model': [self.alg_name, self.alg_name],
            'Dataset': ['Train', 'Test'],
            'Accuracy': [accuracy_train, accuracy_test],
            'F1 Score': [f1_train, f1_test],
            'Precision': [precision_train, precision_test],
            'Recall': [recall_train, recall_test],
            'AUC Score': [auc_score_train, auc_score_test]
        })

        if save_model:
            # Implement model saving logic here, e.g., using joblib or pickle
            pass

        # Plot ROC curve for both test and train data
        self.plot_roc_curve(y_test, probabilities_test, 'Test')
        self.plot_roc_curve(y_train, probabilities_train, 'Train')

        return evaluation_report

    # def plot_roc_curve(self, y_true: np.array, probabilities: np.array, dataset_label: str) -> None:
    #     """Plot the ROC curve for a given dataset."""
    #     fpr, tpr, _ = roc_curve(y_true, probabilities)
    #     roc_auc = auc(fpr, tpr)

    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve ({dataset_label} - area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'Receiver Operating Characteristic - {dataset_label}')
    #     plt.legend(loc="lower right")
    #     plt.show()

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return f"{self.alg_name}_n{self.n_estimators}"


class BalancedBagging(BaseLearningAlgorithm):
    """Balanced Bagging Classifier implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, n_estimators=10, base_estimator=None, random_state=42, n_jobs=-1):
        self.model = BalancedBaggingClassifier(n_estimators=n_estimators, 
                                               estimator=base_estimator, 
                                               random_state=random_state, n_jobs=n_jobs)
        self.alg_name = "BalancedBagging"
        self.n_estimators = n_estimators

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Fit the Balanced Bagging model to the training data."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predict using the fitted Balanced Bagging model."""
        return self.model.predict(x_test)

    def train_eval(self, x_train: pd.DataFrame, y_train: np.array, 
                   x_test: pd.DataFrame, y_test: np.array, x_val: pd.DataFrame = None, 
                   y_val: np.array = None, save_model: bool = True) -> pd.DataFrame:
        """Train and evaluate the Balanced Bagging model."""
        self.fit(x_train, y_train, x_val, y_val)

        predictions_test = self.predict(x_test)
        probabilities_test = self.model.predict_proba(x_test)[:, 1]

        predictions_train = self.predict(x_train)
        probabilities_train = self.model.predict_proba(x_train)[:, 1]

        # Evaluation metrics for test data
        f1_test = f1_score(y_test, predictions_test, zero_division=0)
        accuracy_test = accuracy_score(y_test, predictions_test)
        precision_test = precision_score(y_test, predictions_test, zero_division=0)
        recall_test = recall_score(y_test, predictions_test, zero_division=0)
        auc_score_test = roc_auc_score(y_test, probabilities_test)

        # Evaluation metrics for train data
        f1_train = f1_score(y_train, predictions_train, zero_division=0)
        accuracy_train = accuracy_score(y_train, predictions_train)
        precision_train = precision_score(y_train, predictions_train, zero_division=0)
        recall_train = recall_score(y_train, predictions_train, zero_division=0)
        auc_score_train = roc_auc_score(y_train, probabilities_train)


        evaluation_report = pd.DataFrame({
            'Model': [self.alg_name, self.alg_name],
            'Dataset': ['Train', 'Test'],
            'Accuracy': [accuracy_train, accuracy_test],
            'F1 Score': [f1_train, f1_test],
            'Precision': [precision_train, precision_test],
            'Recall': [recall_train, recall_test],
            'AUC Score': [auc_score_train, auc_score_test]
        })

        if save_model:
            # Implement model saving logic here, e.g., using joblib or pickle
            pass

        self.plot_roc_curve(y_test, probabilities_test, 'Test')
        self.plot_roc_curve(y_train, probabilities_train, 'Train')

        return evaluation_report

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return f"{self.alg_name}_n{self.n_estimators}"
    
class SMOTEBaggingClassifier(BalancedBagging):
    """SMOTE-Bagging Classifier implementation."""

    def __init__(self, n_estimators=10, base_estimator=DecisionTreeClassifier(), random_state=42, n_jobs=-1):
        # Initialize SMOTE with custom parameters if provided
        smote = SMOTE(random_state=random_state)
        # Create a pipeline with SMOTE and the base estimator
        # Use DecisionTreeClassifier as the default base_estimator if none is provided
        base_estimator_pipeline = make_pipeline(smote, base_estimator)
        super().__init__(n_estimators=n_estimators, base_estimator=base_estimator_pipeline, 
                         random_state=random_state, n_jobs=n_jobs)
        self.alg_name = "SMOTEBagging"

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return f"{self.alg_name}_n{self.n_estimators}"