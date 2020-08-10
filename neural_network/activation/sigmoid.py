from neural_network.activation import Activation
import numpy as np


class Sigmoid(Activation):
    def f(self, z: np.ndarray):
        return 1.0 / (1.0 + np.exp(-z))

    def d(self, z: np.ndarray):
        return self.f(z) * (1.0 - self.f(z))
