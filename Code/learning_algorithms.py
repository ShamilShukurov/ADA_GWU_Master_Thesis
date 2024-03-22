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
from ramo import *
from smote import *
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

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_{self.kernel}_C{self.C}"
    


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


    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_{self.booster}_depth{self.max_depth}_lr{self.learning_rate}_est{self.n_estimators}"
    

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


    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_n{self.n_estimators}"


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


    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name#f"{self.alg_name}_n{self.n_estimators}"
    
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


    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name