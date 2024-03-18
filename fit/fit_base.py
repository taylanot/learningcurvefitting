from abc import ABC, abstractmethod

class Fit(ABC):
    @abstractmethod
    def all(self):
        pass
    @abstractmethod
    def id(self):
        pass
    @abstractmethod
    def ids(self):
        pass
