import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseLearningAlgorithm(ABC):
  """Base class for a Supervised Learning Algorithm."""

  @abstractmethod
  def fit(self, x_train:pd.DataFrame, y_train: np.array
          , x_val:pd.DataFrame
          , y_val:np.array) -> None:
    """Trains a model from labels y and examples X.
        Validation set is for optional hyperparameter tuning.
    """

  @abstractmethod
  def predict(self, x_test: pd.DataFrame) -> np.array:
    """Predicts on an unlabeled sample, X."""

  @abstractmethod
  def train_eval(self, x_train:pd.DataFrame, y_train: np.array
                 , x_test:pd.DataFrame , y_test:np.array 
                 , x_val:pd.DataFrame, y_val:np.array
                 , save_model : bool=True
                ) -> pd.DataFrame:
    """Trains the model and return model evaluation report based on train and test data.
        --save_model (bool): Flag to indicate whether to save the trained model.
    """

  @property
  @abstractmethod
  def name(self) -> str:
    """Returns the name of the algorithm."""