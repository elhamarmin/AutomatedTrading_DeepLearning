from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass