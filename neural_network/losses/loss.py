from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def __call__(self) -> float:
        pass

    @abstractmethod
    def derivative(self) -> float:
        pass
