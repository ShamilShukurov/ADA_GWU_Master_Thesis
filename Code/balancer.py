import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class Balancer(ABC):
  """Wrapper class for Data Balancing algorithms."""

  @abstractmethod
  def balance_data(self, x_train:pd.DataFrame, y_train:np.array) -> pd.DataFrame:
    """
        Abstract method to balance the input dataset.
        Parameters:
        - x_train (pd.DataFrame): Input DataFrame containing training data.
        Returns:
        - pd.DataFrame: Balanced DataFrame after applying the balancing algorithm.
    """
  @property
  @abstractmethod
  def name(self) -> str:
    """Returns the name of the algorithm."""



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
    

class RandomUnderSamplerBalancer(Balancer):
    """Random Under Sampler implementation of the Balancer abstract class."""

    def __init__(self):
        self.rus = RandomUnderSampler(random_state=42)
        self._name = "RandomUnderSample"

    def balance_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        X_resampled, y_resampled = self.rus.fit_resample(x_train, y_train)
        # Concatenating the resampled features and target into a single DataFrame
        balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=x_train.columns), pd.Series(y_resampled, name=y_train.name)], axis=1)
        
        return balanced_data

    @property
    def name(self) -> str:
        """Returns the name of the algorithm."""
        return self._name