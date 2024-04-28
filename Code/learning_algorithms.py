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
import lightgbm as lgb
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



class LGBMClf_test(BaseLearningAlgorithm):
    """Logistic Regression implementation of the BaseLearningAlgorithm."""
    
    def __init__(self, alg_name='LGBM'):
        self.model = lgb.LGBMClassifier()
        self.alg_name = alg_name

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Fit the Logistic Regression model to the training data."""
        print(f"len:{len(y_train)} sum:{y_train.sum()}")
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


class ShaDow(BaseLearningAlgorithm):
    def __init__(self, minority_threshold, tree_clf, main_clf):
        self.minority_threshold = minority_threshold  # Threshold to filter majority class
        self.tree_clf = tree_clf                      # Shallow decision tree classifier
        self.main_clf = main_clf                      # Main classifier
    
    def fit(self, X_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None):
        # Fit the shallow tree model
        self.tree_clf.fit(X_train, y_train)
        
        # Get the probabilities of the majority class from the shallow tree
        probs = self.tree_clf.predict_proba(X_train)[:, 0]  # Assuming class 1 is the minority class
        
        # Filter rows where probability of majority class is >= minority_threshold
        filter_mask = probs < self.minority_threshold
        X_filtered = X_train[filter_mask]
        y_filtered = y_train[filter_mask]
        print(f"Shape of the input data {X_train.shape}")
        print(y_train.sum()/len(y_train))
        print(f"Shape of filtered data {X_filtered.shape}")
        print(y_filtered.sum()/len(y_filtered))

        # Fit the main classifier on the filtered data
        self.main_clf.fit(X_filtered, y_filtered)
    
    def predict_proba(self, X_test):
        # Predict probabilities with the shallow tree
        tree_probs = self.tree_clf.predict_proba(X_test)
        
        # Initialize an array to store the final probabilities
        final_probs = np.zeros_like(tree_probs)
        
        # For cases with a high probability of majority class, use tree probabilities
        high_prob_mask = tree_probs[:, 1] >= self.minority_threshold
        final_probs[high_prob_mask] = tree_probs[high_prob_mask]
        
        # For other cases, use probabilities from the main classifier
        low_prob_mask = ~high_prob_mask
        if np.any(low_prob_mask):
            final_probs[low_prob_mask] = self.main_clf.predict_proba(X_test[low_prob_mask])
        
        return final_probs

    def predict(self, X_test):
        # Predict class labels using the shallow tree
        tree_predictions = self.tree_clf.predict(X_test)
        
        # Identify index of the majority class
        # majority_class_index = np.argmin(self.tree_clf.classes_)
        
        # Get probabilities from the shallow tree for the majority class
        tree_probs = self.tree_clf.predict_proba(X_test)[:, 0]
        
        # Initialize an array to store the final predictions
        final_predictions = np.zeros_like(tree_predictions)
        
        # For cases with a high probability of majority class, use tree predictions
        high_prob_mask = tree_probs >= self.minority_threshold
        final_predictions[high_prob_mask] = tree_predictions[high_prob_mask]
        
        # For other cases, use predictions from the main classifier
        low_prob_mask = ~high_prob_mask
        if np.any(low_prob_mask):
            final_predictions[low_prob_mask] = self.main_clf.predict(X_test[low_prob_mask])
        
        return final_predictions
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return "ShaDow"



class SelfBoostingXGBoostClassifier(XGBoostClassifier):
    """Implements the self-boosting approach using two XGBoost models."""

    def __init__(self, alg_name='SelfBoostingXGB', max_depth=3, learning_rate=0.1, 
                 n_estimators=100, verbosity=0, objective='binary:logistic', 
                 booster='gbtree'):

        super().__init__(alg_name=alg_name, max_depth=max_depth, learning_rate=learning_rate, 
                         n_estimators = n_estimators, verbosity=verbosity, objective=objective, 
                         booster=booster, class_weight='balanced')

        # Initialize the second XGBoost model
        self.model_stage2 = xgb.XGBClassifier(alg_name=alg_name, max_depth=max_depth, learning_rate=learning_rate, 
                                              n_estimators=n_estimators, verbosity=verbosity, objective=objective, 
                                              booster=booster, class_weight=None)

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        # First stage: Train with balanced sample weights
        self.model.fit(x_train, y_train)
        
        # Generate probabilities from the first model
        probabilities_train = self.model.predict(x_train)#[:, 1]
        
        # Add probabilities as a feature for the second training stage
        x_train_with_probs = x_train.copy()
        x_train_with_probs['probabilities'] = probabilities_train
        
        # Second stage: Train the second model without class weighting
        self.model_stage2.fit(x_train_with_probs, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        # Use the second model for prediction
        probabilities_test = self.model.predict(x_test)#[:, 1]
        x_test_with_probs = x_test.copy()
        x_test_with_probs['probabilities'] = probabilities_test
        return self.model_stage2.predict(x_test_with_probs)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        # Predict probabilities using the second model
        probabilities_test = self.model.predict(x_test)#[:, 1]
        x_test_with_probs = x_test.copy()
        x_test_with_probs['probabilities'] = probabilities_test
        return self.model_stage2.predict_proba(x_test_with_probs)

class PSMClassifier(BaseLearningAlgorithm):
    """
    Parallel Stacked Models Classifier that uses predictions from multiple imbalance-handling models
    as features for a final main model.
    """
    def __init__(self, sub_models: list, main_model: BaseLearningAlgorithm):
        """
        Initializes the PSMClassifier with a list of sub-models and a main model.
        
        :param sub_models: List of instances of BaseLearningAlgorithm handling imbalance.
        :param main_model: The main model instance of BaseLearningAlgorithm to make the final prediction.
        """
        self.sub_models = sub_models
        self.main_model = main_model

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        # First, fit all sub-models
        sub_model_preds = []
        for model in self.sub_models:
            print(f"Fitting the model {model.name}...")
            model.fit(x_train, y_train, x_val, y_val)
            # Collect predictions as additional features
            preds = model.predict(x_train)#[:, 1]  # assuming binary classification
            sub_model_preds.append(preds)
        
        # Create a new training set for the main model that includes predictions from sub-models
        x_train_with_preds = x_train.copy()
        for i, preds in enumerate(sub_model_preds):
            x_train_with_preds[f'P_{self.sub_models[i].name}'] = preds
        
        # Fit the main model on the new training set
        self.main_model.fit(x_train_with_preds, y_train, x_val, y_val)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        # First, get predictions from all sub-models
        sub_model_preds = []
        for model in self.sub_models:
            preds = model.predict(x_test)#[:, 1]  # assuming binary classification
            sub_model_preds.append(preds)
        
        # Create a new test set that includes these predictions
        x_test_with_preds = x_test.copy()
        for i, preds in enumerate(sub_model_preds):
            x_test_with_preds[f'P_{self.sub_models[i].name}'] = preds
        
        # Use the main model to make final predictions on the enhanced test set
        return self.main_model.predict(x_test_with_preds)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        # Similar to predict, but return probabilities
        sub_model_preds = []
        for model in self.sub_models:
            preds = model.predict(x_test)#[:, 1]  # assuming binary classification
            sub_model_preds.append(preds)
        
        x_test_with_preds = x_test.copy()
        for i, preds in enumerate(sub_model_preds):
            x_test_with_preds[f'P_{self.sub_models[i].name}'] = preds
        
        return self.main_model.predict_proba(x_test_with_preds)
    
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return "PSM"

class SSMClassifier(BaseLearningAlgorithm):
    """
    Sequentially Stacked Models Classifier that sequentially trains models,
    each using the original features augmented by the predictions of the previously trained models.
    """
    def __init__(self, models: list):
        """
        Initializes the SSMClassifier with a list of models to be trained sequentially.
        
        :param models: List of instances of BaseLearningAlgorithm.
        """
        self.models = models

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        # Current feature set, starting with the original features
        current_x_train = x_train.copy()

        for model in self.models:
            print(f"Fitting the model {model.name}...")
            # Train the current model on the current feature set
            model.fit(current_x_train, y_train, x_val, y_val)
            # Predict probabilities and add as a new feature for the next model
            preds = model.predict(current_x_train)#[:, 1]  # assuming binary classification
            current_x_train[f'{model.name}_preds'] = preds

    def predict(self, x_test: pd.DataFrame) -> np.array:
        # Start with original test set
        current_x_test = x_test.copy()
        
        # Sequentially predict and augment the feature set
        for model in self.models:
            preds = model.predict(current_x_test)#[:, 1]
            current_x_test[f'{model.name}_preds'] = preds

        # Final prediction using the last model
        return self.models[-1].predict(current_x_test)

    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        # Similar to predict, but return probabilities using the last model
        current_x_test = x_test.copy()
        
        for model in self.models:
            preds = model.predict(current_x_test)#[:, 1]
            current_x_test[f'{model.name}_preds'] = preds

        return self.models[-1].predict_proba(current_x_test)
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return "SSM"


class DCBoost(BaseLearningAlgorithm):
    """
    Two-stage XGBoost classifier that first maximizes recall by giving high weight to the minority class,
    and then refines predictions by filtering and retraining on predicted positives.
    """
    def __init__(self, min_weight=95, max_depth=3, learning_rate=0.1, 
                 n_estimators=100, verbosity=0, objective='binary:logistic', 
                 booster='gbtree', second_model_balanced = True):
        """
        Initializes the TwoStageXGBoost classifier with custom weights and additional XGBoost parameters.
        
        :param high_weight: The weight to assign to the minority class in the first model to maximize recall.
        """
        self.min_weight = min_weight
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.objective = objective
        self.booster = booster
        self.second_model_balanced = second_model_balanced
        self.model_stage1 = xgb.XGBClassifier(max_depth=max_depth, 
                                       learning_rate=learning_rate, 
                                       n_estimators=n_estimators, 
                                       verbosity=verbosity, 
                                       objective=objective, 
                                       booster=booster, 
                                       use_label_encoder=False)

        self.model_stage2 = xgb.XGBClassifier(max_depth=max_depth, 
                                       learning_rate=learning_rate, 
                                       n_estimators=n_estimators, 
                                       verbosity=verbosity, 
                                       objective=objective, 
                                       booster=booster, 
                                       use_label_encoder=False)




    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:

        self.model_stage1 = xgb.XGBClassifier(max_depth=self.max_depth, 
                                learning_rate=self.learning_rate, 
                                n_estimators=self.n_estimators, 
                                verbosity=self.verbosity, 
                                objective=self.objective, 
                                booster=self.booster, 
                                use_label_encoder=False,
                                scale_pos_weight = max(get_scale_weight(y_train),self.min_weight))

        # Train the first model on all data with a high weight for the minority
        self.model_stage1.fit(x_train, y_train, eval_metric='logloss')

        # Predict on training set and filter instances where prediction is 1
        train_preds = self.model_stage1.predict(x_train)
        x_train_filtered = x_train[train_preds == 1]
        y_train_filtered = y_train[train_preds == 1]

        if self.second_model_balanced:
            one_count = len(y_train_filtered[y_train_filtered==1])
            zero_count = len(y_train_filtered[y_train_filtered==0])
            w = zero_count/one_count
            self.model_stage2 = xgb.XGBClassifier(max_depth=self.max_depth, 
                                    learning_rate=self.learning_rate, 
                                    n_estimators=self.n_estimators, 
                                    verbosity=self.verbosity, 
                                    objective=self.objective, 
                                    booster=self.booster, 
                                    use_label_encoder=False,
                                    scale_pos_weight = w)

        print(y_train_filtered.value_counts())
        print("============================")
        # Train the second model on the filtered dataset
        if len(y_train_filtered) > 0:  # Check if there are any samples to train on
            self.model_stage2.fit(x_train_filtered, y_train_filtered, eval_metric='logloss')

    def predict(self, x_test: pd.DataFrame) -> np.array:
        # First stage to filter instances
        test_preds = self.model_stage1.predict(x_test)
        x_test_filtered = x_test[test_preds == 1]

        # Initialize with zeros for all instances
        final_predictions = np.zeros(x_test.shape[0], dtype=int)

        # Predict with the second model on filtered instances
        if len(x_test_filtered) > 0:  # Check if there are any samples to predict
            filtered_predictions = self.model_stage2.predict(x_test_filtered)
            final_predictions[test_preds == 1] = filtered_predictions

        return final_predictions


    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        # First model to predict probabilities
        proba_first_model = self.model_stage1.predict_proba(x_test)

        # Filter instances based on predictions of the first model
        predicted_ones = self.model_stage1.predict(x_test) == 1
        x_test_filtered = x_test[predicted_ones]

        # Initialize with first model probabilities for all instances
        final_probabilities = proba_first_model

        # Predict probabilities for filtered instances using the second model
        if len(x_test_filtered) > 0:  # Check if there are any samples to predict
            filtered_probabilities = self.model_stage2.predict_proba(x_test_filtered)
            final_probabilities[predicted_ones, :] = filtered_probabilities

        return final_probabilities

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return "DCBoost"