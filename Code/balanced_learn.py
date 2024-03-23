from base_learning import BaseLearningAlgorithm
from balancer import Balancer
import pandas as pd
import numpy as np


class BalancedLearn(BaseLearningAlgorithm):
    """Integrates a Balancer with a Learning Algorithm."""

    def __init__(self, balancer:Balancer, learning_algorithm:BaseLearningAlgorithm):
        """
        Initializes the BalancedLearn class with specified balancer and learning algorithm.
        
        Parameters:
        - balancer (Balancer): An instance of a class extending the Balancer abstract class.
        - learning_algorithm (BaseLearningAlgorithm): An instance of a class extending the BaseLearningAlgorithm class.
        """
        self.balancer = balancer
        self.learning_algorithm = learning_algorithm
        self.alg_name = f"{balancer.name}_{learning_algorithm.name}"

    def fit(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame = None, y_val: np.array = None) -> None:
        """Balances the training data and fits the learning algorithm to the balanced dataset."""
        # Balance the training data
        balanced_data = self.balancer.balance_data(x_train, y_train)
        # Split balanced_data back into features and target
        y_balanced = balanced_data.iloc[:, -1].values
        x_balanced = balanced_data.drop(balanced_data.columns[-1], axis=1)
        
        # Fit the learning algorithm to the balanced dataset
        self.learning_algorithm.fit(x_balanced, y_balanced, x_val, y_val)

    def predict(self, x_test: pd.DataFrame) -> np.array:
        """Predicts on the test data. Optionally balances the test data if specified."""
        if self.balancer.apply_to_test:
            # Balance the test data similar to the training data
            balanced_test_data = self.balancer.balance_data(x_test)
            # print(balanced_test_data.columns)
            return self.learning_algorithm.predict(balanced_test_data)

        return self.learning_algorithm.predict(x_test)
    def predict_proba(self, x_test: pd.DataFrame) -> np.array:
        """Predicts on the test data. Optionally balances the test data if specified."""
        if self.balancer.apply_to_test:
            # Balance the test data similar to the training data
            balanced_test_data = self.balancer.balance_data(x_test)
            return self.learning_algorithm.predict_proba(balanced_test_data)
        
        return self.learning_algorithm.predict_proba(x_test)
    
    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.alg_name
