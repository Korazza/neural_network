from neural_network.activation import Activation
from neural_network.layers import Layer
from typing import Union
import numpy as np


id_counter = 1


class Dense(Layer):
    def __init__(
        self,
        units: int,
        input_dim: int = None,
        activation: Union[str, Activation] = "linear",
    ):
        Layer.__init__(self, units, input_dim, activation)
        global id_counter
        self.set_id(id_counter)
        id_counter += 1
        self.input_dim = input_dim
        if input_dim:
            self.init_weights()
        self.biases = np.zeros((1, units))

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.init_weights()

    def init_weights(self):
        self.weights = np.random.randn(self.input_dim, self.units) * np.sqrt(
            2 / (self.input_dim + self.units)
        )

    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation(self.z)
        return self.output
