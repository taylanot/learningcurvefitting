"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : fit.py
"""
import time 
import numpy as np

def SquaredError(model, curve):
        return (model.predict(curve.anchors)-curve.labels)**2

def fit(curves,model,till):
    """ 
        Fit all the data
        curves  : MxN M full curves with N acnhors
        model   : which model used to fit the curve
        till    : number of anchors from the  start to fit
    """
    error = np.zeros((curves.M,curves.N))
    status = np.empty(shape=(curves.M),dtype=np.str_)
    init = np.zeros((curves.M, model.ntheta))
    final = np.zeros((curves.M, model.ntheta))
    fit_sec = np.zeros(curves.M)
    #try:
    #    from tqdm import tqdm
    #    with tqdm(total=curves.M) as progress_bar:
    #        for i, curve in enumerate(tqdm(curves)):
    #            _, status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
    #                    fit_id(curve,model,till) 
    #except:
    #    for i, curve in enumerate(curves):
    #        _, status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
    #                fit_id(curve,model,till) 
    for i, curve in enumerate(curves):
        try:
            _, status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
                fit_idx(curve,model,till,i) 
        except KeyboardInterrupt:
            break
        except:
            print("fit_failed")

    return {"curve_tags":curves.tags,
            "status":status, "theta0":init, "theta":final,\
            "error":error, "time":fit_sec}

def fit_idx(curves,model,till,idx):
    """ 
        curve   : 1xN shaped M full curves
        model   : which model used to fit the curve 
        till    : number of anchors from the  start to fit 
        idx     : id of the curve
    """
    res = fit_one(curves[idx],model,till)
    return res.values()
          
def fit_one(curve,model,till):
    """ 
        curve   : 1xN shaped M full curves
        model   : which model used to fit the curve 
        till    : number of anchors from the  start to fit 
    """
    start = time.time()
    model.fit(curve[:till])
    model.predict(curve)
    end = time.time()
    if model.status:
        status = "s"
    else:
        status = "f"
    return {"curve_tag":curve.tag, "status":status,
            "theta0":model._theta0, "theta":model.theta, \
                    "error":SquaredError(model,curve), "time":end-start}

