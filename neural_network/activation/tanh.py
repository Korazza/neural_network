from neural_network.activation import Activation
import numpy as np


class Tanh(Activation):
    def f(self, z: np.ndarray):
        return np.tanh(z)

    def d(self, z: np.ndarray):
        return 1.0 - self.f(z) ** 2
