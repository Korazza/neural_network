from neural_network.losses import Loss
import numpy as np


class MeanSquaredError(Loss):
    def f(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.sum(np.square(y_true - y_pred)) / y_true.size

    def d(self, y_true: np.ndarray, y_pred: np.ndarray):
        return -2 * np.sum(y_true - y_pred)
