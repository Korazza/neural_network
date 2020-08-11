from neural_network.activation import Activation
import numpy as np


class Tanh(Activation):
    def __call__(self, z: np.ndarray):
        return np.tanh(z)

    def derivative(self, z: np.ndarray):
        return 1.0 - self(z) ** 2
