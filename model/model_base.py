from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def get_exact_gradients(self,x):
        pass
    @abstractmethod
    def get_approx_gradients(self,x):
        pass
    @abstractmethod
    def objective(self,data):
        pass
    @abstractmethod
    def optimize(self,data):
        pass
