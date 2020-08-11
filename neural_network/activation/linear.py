from neural_network.activation import Activation
import numpy as np


class Linear(Activation):
    def __call__(self, z: np.ndarray):
        return z

    def derivative(self, z: np.ndarray):
        return np.ones(z.shape)
