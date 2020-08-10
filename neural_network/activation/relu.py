from neural_network.activation import Activation
import numpy as np


class Relu(Activation):
    def f(self, z: np.ndarray):
        return np.maximum(0, z)

    def d(self, z: np.ndarray):
        return np.greater(z, 0).astype(float)
