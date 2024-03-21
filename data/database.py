"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : data/database.py
"""

from abc import ABC, abstractmethod, abstractproperty

class curve(ABC):
    @abstractproperty
    def anchor(self):
        pass
    @abstractproperty
    def labels(self):
        pass
    # Generator for the points on the curve

class LCDB():
    @abstractmethod
    def get_train(self,idx):
        pass
    @abstractmethod
    def get_valid(self,idx):
        pass
    @abstractmethod
    def get_test(self,idx):
        pass

    # Generator for the curves in the database

