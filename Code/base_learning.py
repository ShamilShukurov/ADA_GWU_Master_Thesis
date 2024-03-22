import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, roc_auc_score, roc_curve, auc

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
  def plot_roc_curve(self, y_true: np.array, probabilities: np.array, dataset_label: str) -> None:
      """Plot the ROC curve for a given dataset."""
      fpr, tpr, _ = roc_curve(y_true, probabilities)
      roc_auc = auc(fpr, tpr)

      plt.figure()
      plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve ({dataset_label} - area = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(f'Receiver Operating Characteristic - {dataset_label}')
      plt.legend(loc="lower right")
      plt.show()
  @property
  @abstractmethod
  def name(self) -> str:
    """Returns the name of the algorithm."""