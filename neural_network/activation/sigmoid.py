from neural_network.activation import Activation
import numpy as np


class Sigmoid(Activation):
    def __call__(self, z: np.ndarray):
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z: np.ndarray):
        return self(z) * (1.0 - self(z))
