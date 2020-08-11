from neural_network.activation import Activation
import numpy as np


class Relu(Activation):
    def __call__(self, z: np.ndarray):
        return np.maximum(0, z)

    def derivative(self, z: np.ndarray):
        return np.greater(z, 0).astype(float)
