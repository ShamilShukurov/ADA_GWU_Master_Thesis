import pandas as pd
from abc import ABC, abstractmethod

class Balancer(ABC):
  """Wrapper class for Data Balancing algorithms."""

  @abstractmethod
  def balance_data(self, x_train:pd.DataFrame) -> pd.DataFrame:
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