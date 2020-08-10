from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def f(self) -> float:
        pass

    @abstractmethod
    def d(self) -> float:
        pass
