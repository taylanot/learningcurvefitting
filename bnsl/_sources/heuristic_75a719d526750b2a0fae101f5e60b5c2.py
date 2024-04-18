"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : model/heuristic.py
"""
class LAST1():
    def __init__(self,config):
        self.config = config
    def fit(self,curve):
        self.theta = curve.labels[-1]
    def predict(self,xs):
        return self.theta

class LASTGRAD():
    def __init__(self,config):
        self.config = config
    def fit(self,curve):
        self.theta = (curve.labels[-1]-curve.labels[-2]) / \
                (curve.anchors[-1]-curve.anchors[-2])
    def predict(self,xs):
        return self.theta*xs

