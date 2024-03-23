import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

class Balancer(ABC):
  """Wrapper class for Data Balancing algorithms."""

  @abstractmethod
  def balance_data(self, x_train:pd.DataFrame, y_train:np.array) -> pd.DataFrame:
      """
        Abstract method to balance the input dataset.
      
      Parameters:
      - x_train (pd.DataFrame): Input DataFrame containing features.
      - y_train (pd.Series): Input Series containing the target variable.
      
      Returns:
      - pd.DataFrame: Balanced DataFrame after applying balancing algorithm.
      """
  @property
  @abstractmethod
  def name(self) -> str:
    """Returns the name of the algorithm."""

  @property
  @abstractmethod
  def apply_to_test(self) -> bool:
    """Whether this technique should be applied to test set or not"""


class RandomOverSamplerBalancer(Balancer):
    """Random Over Sampler implementation of the Balancer abstract class."""

    def __init__(self):
        self.ros = RandomOverSampler(random_state=42)
        self._name = "RandomOverSample"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:

        X_resampled, y_resampled = self.ros.fit_resample(x_train, y_train)
        return pd.concat([pd.DataFrame(X_resampled, columns=x_train.columns), 
                          pd.Series(y_resampled)],axis=1)

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name
    
    @property
    def apply_to_test(self) -> bool:
        return False
    

class RandomUnderSamplerBalancer(Balancer):
    """Random Under Sampler implementation of the Balancer abstract class."""

    def __init__(self):
        self.rus = RandomUnderSampler(random_state=42)
        self._name = "RandomUnderSample"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        X_resampled, y_resampled = self.rus.fit_resample(x_train, y_train)
        # Concatenating the resampled features and target into a single DataFrame
        balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=x_train.columns), 
                                   pd.Series(y_resampled, name=y_train.name)], axis=1)
        
        return balanced_data

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name
    
    @property
    def apply_to_test(self) -> bool:
        return False    

class SMOTEBalancer(Balancer):
    """SMOTE implementation of the Balancer abstract class."""

    def __init__(self, sampling_strategy='auto', random_state=42):
        self.smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        self._name = "SMOTE"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        X_resampled, y_resampled = self.smote.fit_resample(x_train, y_train)
        # Concatenating the resampled features and target into a single DataFrame
        balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=x_train.columns), 
                                   pd.Series(y_resampled, name=y_train.name)], axis=1)
        
        return balanced_data

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name
    
    @property
    def apply_to_test(self) -> bool:
        return False

class ADASYNBalancer(Balancer):
    """ADASYN implementation of the Balancer abstract class."""

    def __init__(self, sampling_strategy='auto', random_state=42):
        self.adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        self._name = "ADASYN"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        X_resampled, y_resampled = self.adasyn.fit_resample(x_train, y_train)
        # Concatenating the resampled features and target into a single DataFrame
        balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=x_train.columns), pd.Series(y_resampled, name=y_train.name)], axis=1)
        
        return balanced_data

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name

    @property
    def apply_to_test(self) -> bool:
        return False
        
class TomekLinksBalancer(Balancer):
    """Tomek Links implementation of the Balancer abstract class."""

    def __init__(self):
        self.tomek = TomekLinks()
        self._name = "TomekLinks"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        X_resampled, y_resampled = self.tomek.fit_resample(x_train, y_train)
        # Concatenating the resampled features and target into a single DataFrame
        balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=x_train.columns), pd.Series(y_resampled, name=y_train.name)], axis=1)
        
        return balanced_data

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name

    @property
    def apply_to_test(self) -> bool:
        return False    

class SMOTETomekBalancer(Balancer):
    """SMOTE-TOMEK implementation of the Balancer abstract class."""

    def __init__(self, sampling_strategy='auto', random_state=42):
        self.smote_tomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
        self._name = "SMOTETomek"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        X_resampled, y_resampled = self.smote_tomek.fit_resample(x_train, y_train)
        # Concatenating the resampled features and target into a single DataFrame
        balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=x_train.columns), pd.Series(y_resampled, name=y_train.name)], axis=1)
        
        return balanced_data

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name

    @property
    def apply_to_test(self) -> bool:
        return False    

class LOFEnhance(Balancer):
    """Local Outlier Factor implementation of the Balancer abstract class."""
    
    def __init__(self, n_neighbors=20, contamination='auto'):
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        self._name = "LocalOutlierFactor"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series = None) -> pd.DataFrame:
        """Note y_train can be None only for enhancers. 
           That is because we apply enhancing on test set as well where we don't need y_test
        """

        # Fit the LOF model to the data and predict the outlier status directly
        outlier_predictions = self.lof.fit_predict(x_train)

        # Transform predictions to match our desired format: 1 for outliers, 0 for inliers
        is_outlier = np.where(outlier_predictions == -1, 1, 0)
        
        # Compute anomaly scores (the negative outlier factor)
        lof_scores = self.lof.negative_outlier_factor_
        # Invert scores to make them more intuitive (higher means more outlier-like)
        normalized_scores = -lof_scores
        
        # Add LOF scores and outlier indicators to the dataset
        enhanced_data = x_train.copy()
        enhanced_data['LOF_Score'] = normalized_scores
        enhanced_data['Is_Outlier_LOF'] = is_outlier
        if y_train is not None:
          enhanced_data[y_train.name] = y_train
        return enhanced_data

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name
    
    @property
    def apply_to_test(self) -> bool:
        return True
    
class IsolationForestEnhance(Balancer):
    """Isolation Forest Balancer implementation of the Balancer abstract class."""
    
    def __init__(self, n_estimators=100, contamination='auto', random_state=42):
        self.isoforest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
        self._name = "IsolationForest"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series = None) -> pd.DataFrame:
        """Note y_train can be None only for enhancers. 
           That is because we apply enhancing on test set as well where we don't need y_test
        """
        # Fit the Isolation Forest model to the data
        self.isoforest.fit(x_train)
        # Predict the outlier status (-1 for outliers, 1 for inliers)
        outlier_predictions = self.isoforest.predict(x_train)
        # Compute anomaly scores (lower scores indicate more outlier-like)
        anomaly_scores = self.isoforest.decision_function(x_train)
        
        # Transform predictions to match our desired format: 1 for outliers, 0 for inliers
        is_outlier = np.where(outlier_predictions == -1, 1, 0)
        # Normalize anomaly scores to be positive (higher means more outlier-like)
        normalized_scores = -anomaly_scores
        
        # Add anomaly scores and outlier indicators to the dataset
        enhanced_data = x_train.copy()
        enhanced_data['IsoForest_Score'] = normalized_scores
        enhanced_data['Is_Outlier_IF'] = is_outlier
        if y_train is not None:
          enhanced_data[y_train.name] = y_train
        
        return enhanced_data

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name
    
    @property
    def apply_to_test(self) -> bool:
        return True