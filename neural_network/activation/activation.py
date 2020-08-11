from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self) -> np.ndarray:
        pass
