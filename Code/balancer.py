import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from imblearn.over_sampling import RandomOverSampler


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