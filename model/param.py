"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : model/param.py
"""

from abc import ABC, abstractmethod, abstractproperty

class ParametricModels(ABC):
    @abstractproperty
    def theta(self):
        """
            Return the parameters of the model 
        """
        return self._theta
    @theta.setter
    def theta(self, values):
        """
            Return the parameters of the model 
            value   : 1xK array of parameters 
        """
        self._theta = values 
    @abstractmethod
    def get_exact_gradients(self,xs):
        """
            Return exact gradients at
            xs  : 1xN anchors
        """
        pass
    @abstractmethod
    def get_approx_gradients(self,xs):
        """
            Return approximate gradients at
            xs  : 1xN anchors
        """
        pass
    @abstractmethod
    def objective(self,curve):
        """
            Return objective function to be optimized for
            curve   : object returned from our database
        """
        pass

    @abstractmethod
    def fit(self,curve):
        """
            Fit to the 
            curve   : object returned from our database
        """
        pass
    @abstractmethod
    def predict(self,xs):
        """
            Predict with the model at
            xs  : 1xN anchors
        """
        pass


class Last1():
    def __init__(self,config):
        (ParametricModels,self).__init__(config)
    def fit(self,curve):
        self.parameter = curve[1][-1]
    def predict(self,curve):
        return self.parameter

