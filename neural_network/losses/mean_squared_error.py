from neural_network.losses import Loss
import numpy as np


class MeanSquaredError(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.sum(np.square(y_true - y_pred)) / y_true.size

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        return -2.0 * np.sum(y_true - y_pred)
