"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : fit.py
"""
import time 
import numpy as np

def SquaredError(model, curve):
        return (model.predict(curve.anchors)-curve.labels)**2

def fit(curves,model,at):
    """ 
        Fit all the data
        curves  : MxN M full curves with N acnhors
        model   : which model used to fit the curve
        at      : number of anchors from the  start to fit
    """
    error = np.zeros((curves.M,curves.N))
    status = np.empty(shape=(curves.M),dtype=np.bool_)
    init = np.zeros((curves.M, model.ntheta))
    final = np.zeros((curves.M, model.ntheta))
    fit_sec = np.zeros(curves.M)
    for i, curve in enumerate(curves):
        status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
                fit_id(curve,model,at) 
    return {"status":status, "theta0":init, "theta":final,\
            "error":error, "time":fit_sec}

def fit_id(curve,model,at):
    """ 
        curve   : 1xN shaped M full curves
        model   : which model used to fit the curve 
        at      : number of anchors from the  start to fit 
    """
    start = time.time()
    model.fit(curve[:at])
    model.predict(curve)
    end = time.time()
    return model.status, model._theta0, model.theta, \
            SquaredError(model,curve), end-start

