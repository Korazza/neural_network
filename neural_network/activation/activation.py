from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def f(self) -> np.ndarray:
        pass

    @abstractmethod
    def d(self) -> np.ndarray:
        pass
