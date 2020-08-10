from neural_network.activation import Activation
import numpy as np


class Linear(Activation):
    def f(self, z: np.ndarray):
        return z

    def d(self, z: np.ndarray):
        return np.ones(z.shape)
