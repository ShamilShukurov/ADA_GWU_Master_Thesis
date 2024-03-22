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

  # @abstractmethod
  def train_eval(self, x_train:pd.DataFrame, y_train: np.array
                 , x_test:pd.DataFrame , y_test:np.array 
                 , x_val:pd.DataFrame=None, y_val:np.array=None
                 , save_model : bool=True
                ) -> pd.DataFrame:
    """Trains the model and return model evaluation report based on train and test data.
        --save_model (bool): Flag to indicate whether to save the trained model.
    """

  # def train_eval(self, x_train: pd.DataFrame, y_train: np.array, x_test: pd.DataFrame, y_test: np.array, x_val: pd.DataFrame = None, y_val: np.array = None, save_model: bool = True) -> pd.DataFrame:
  #     """Train and evaluate the XGBoost model on both training and test datasets."""
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