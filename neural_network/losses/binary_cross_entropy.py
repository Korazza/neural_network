from neural_network.losses import Loss
import numpy as np


class BinaryCrossEntropy(Loss):
    def f(self, y_true: np.ndarray, y_pred: np.ndarray):
        return (-1.0 / y_true.size) * np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def d(self, y_true: np.ndarray, y_pred: np.ndarray):
        return (-1.0 / y_true.size) * ((y_true / y_pred) - ((1.0 - y_true) / (1.0 - y_pred)))
