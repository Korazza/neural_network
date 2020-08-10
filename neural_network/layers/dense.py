from neural_network.activation import Activation
from neural_network.layers import Layer
from typing import Union
import numpy as np


id_counter = 1


class Dense(Layer):
    def __init__(
        self,
        units: int,
        input_shape: int = None,
        activation: Union[str, Activation] = "linear",
    ):
        Layer.__init__(self, units, input_shape, activation)
        global id_counter
        self.set_id(id_counter)
        id_counter += 1
        self.input_shape = input_shape
        if input_shape:
            self.init_weights()
        self.biases = 0.01 * np.random.randn(1, units)

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.init_weights()

    def init_weights(self):
        self.weights = 0.01 * np.random.randn(self.input_shape, self.units)

    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.f(self.z)
        return self.output
