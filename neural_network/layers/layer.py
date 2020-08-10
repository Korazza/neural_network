from neural_network.activation import Activation
from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class Layer(ABC):
    def __init__(
        self,
        units: int,
        input_shape: int = None,
        activation: Union[str, Activation] = "linear",
    ):
        self.units = units
        self.activation = (
            getattr(getattr(__import__("neural_network"), "activation"), activation.title())()
            if isinstance(activation, str)
            else activation
        )

    def set_id(self, id_count: int):
        self.id = self.__class__.__name__.lower() + "_" + str(id_count)

    def summary(self):
        print("\n ********* Layer {} *********".format(self.id))
        print("\nUnits:", self.units)
        print("\nWeights:\n", self.weights)
        print("\nBiases:\n", self.biases)
        print("\nActivation:", self.activation.__class__.__name__)

    @abstractmethod
    def set_input_shape(self):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self) -> np.ndarray:
        pass
