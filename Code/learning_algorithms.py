"""
Implementation of different supervised learning algorithms.


All learning algorithms are implemented by extending base wrapper class BaseLearningAlgorithm.
List of implemented algorithms:
        1.SVMClassifier
        2.RandomForest
        3.XGBoostClassifier
        4.LogisticRegressionClassifier
        5.EasyEnsemble
        6.BalancedBagging
        7.SMOTEBagging
        8.RAMOBoost
        9.SMOTEBoost
        10.RUSBoost
        11.IsolationForest
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from ramo import RAMOBoost
from smote import SMOTEBoost
# from rus import RUSBoost
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from base_learning import BaseLearningAlgorithm
from os.path import join
from learning_algorithms import *
from utils import *


class SVMClassifier(BaseLearningAlgorithm):
    """SVM Classifier implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name = 'SVC', kernel='rbf', C=5.0, class_weight=None, verbose=False):
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

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(x_test)
    
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_{self.kernel}_C{self.C}"
    
class RandomForest(BaseLearningAlgorithm):
    """Random Forest Classifier implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name='RandomForest', n_estimators=55, class_weight=None, random_state=42, verbose=False):
        self.model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, 
                                            max_depth=4, random_state=random_state, verbose=verbose)
        self.alg_name = alg_name
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        self.verbose = verbose
        
    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Fit the Random Forest model to the training data."""
        self.model.fit(x_train, y_train)
    
    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predict using the fitted Random Forest model."""
        return self.model.predict(x_test)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        """Predicts probabilities on an unlabeled sample, X."""
        return self.model.predict_proba(x_test)
    
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name

class XGBoostClassifier(BaseLearningAlgorithm):
    """XGBoost Classifier implementation of the BaseLearningAlgorithm."""

    def __init__(self, alg_name='XGB', max_depth=3, learning_rate=0.1, 
                 n_estimators=100, verbosity=0, objective='binary:logistic', 
                 booster='gbtree', class_weight=None):
        self.model = xgb.XGBClassifier(max_depth=max_depth, 
                                       learning_rate=learning_rate, 
                                       n_estimators=n_estimators, 
                                       verbosity=verbosity, 
                                       objective=objective, 
                                       booster=booster, 
                                       use_label_encoder=False)
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
        #if balanced option selected, recreate class model with balanced scale pos weight
        if(self.class_weight == 'balanced'):
            self.model = xgb.XGBClassifier(max_depth=self.max_depth, 
                                learning_rate=self.learning_rate, 
                                n_estimators=self.n_estimators, 
                                verbosity=self.verbosity, 
                                objective=self.objective, 
                                booster=self.booster, 
                                use_label_encoder=False,
                                scale_pos_weight = get_scale_weight(y_train))
            
        self.model.fit(x_train, y_train, eval_metric='logloss')

    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predict using the fitted XGBoost model."""
        return self.model.predict(x_test)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(x_test)
    
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_{self.booster}_depth{self.max_depth}_lr{self.learning_rate}_est{self.n_estimators}"
    

class LogisticRegressionClassifier(BaseLearningAlgorithm):
    """Logistic Regression implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name='Logistic', 
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

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(x_test)

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_{self.penalty}_C{self.C}"


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

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(x_test)

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_n{self.n_estimators}"


class BalancedBagging(BaseLearningAlgorithm):
    """Balanced Bagging Classifier implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, n_estimators=10, base_estimator=DecisionTreeClassifier(max_depth=3), 
                 random_state=42, n_jobs=-1):
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

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(x_test)

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_n{self.n_estimators}"
    
class SMOTEBaggingClassifier(BalancedBagging):
    """SMOTE-Bagging Classifier implementation."""

    def __init__(self, n_estimators=10, base_estimator=DecisionTreeClassifier(max_depth=3), random_state=42, n_jobs=-1):
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
        return self.alg_name#f"{self.alg_name}_n{self.n_estimators}"
    

class RAMOBoostClassifier(BaseLearningAlgorithm):
    """RamoBoost implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name='RAMOBoost'):
        self.model = RAMOBoost()
        self.alg_name = alg_name

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict(x_test)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(x_test)

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name
    
class SMOTEBoostClassifier(BaseLearningAlgorithm):
    """SMOTEBoost implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name='SMOTEBoost'):
        self.model = SMOTEBoost()
        self.alg_name = alg_name

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        self.model.fit(x_train, y_train)


    def predict(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict(x_test)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(x_test)
    
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name
    
class RUSBoost(BaseLearningAlgorithm):
    """SMOTEBoost implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name='RUSBoost'):
        self.model = RUSBoostClassifier()
        self.alg_name = alg_name

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict(x_test)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        return self.model.predict_proba(x_test)
    
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name
    

class IsolationForestClassifier(BaseLearningAlgorithm):
    """Isolation Forest Classifier implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, n_estimators=100, contamination='auto', random_state=42):
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
        self.alg_name = "IsolationForest"
        self.n_estimators = n_estimators
        self.contamination = contamination

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Fit the Isolation Forest model to the training data."""
        # Isolation Forest is designed for outlier detection, thus only X is used for fitting
        self.model.fit(x_train)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predict using the fitted Isolation Forest model."""
        # Isolation Forest predicts -1 for outliers and 1 for inliers. Here, we map these to a binary classification context.
        predictions = self.model.predict(x_test)
        # Map Isolation Forest's -1 (outlier) and 1 (inlier) to 0 and 1 respectively for compatibility with binary classification tasks
        return np.where(predictions == 1, 0, 1)
    
    def decision_function(self, x_test: pd.DataFrame) -> np.array:
        """Compute the anomaly score for each sample."""
        return self.model.decision_function(x_test)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        """Predict probability-like outputs for the fitted IsolationForest model."""
        # Get decision function scores
        decision_scores = self.model.decision_function(x_test)
        # Apply sigmoid function to transform scores into probability-like values
        probability_like = 1 / (1 + np.exp(-decision_scores))
        # Since we're dealing with a binary classification (inlier vs outlier), 
        # we need probabilities for both classes. We can consider the probability of being an outlier
        # as the transformed score, and the probability of being an inlier as 1 minus this value.
        return np.vstack((probability_like,1 - probability_like)).T

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name