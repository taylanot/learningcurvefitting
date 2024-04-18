"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : fit.py
"""
import time 
import numpy as np
def SquaredError(model, curve):
        #import matplotlib.pyplot as plt
        #plt.plot(curve.anchors,curve.labels,'r--')
        #plt.plot(curve.anchors,model.predict(curve.anchors),'k')
        #plt.show()
        return (model.predict(curve.anchors)-curve.labels)**2

def fit(curves,model,till,idx=None):
    """ 
        Fit all the data
        curves  : MxN M full curves with N acnhors
        model   : which model used to fit the curve
        till    : number of anchors from the  start to fit
    """
    if idx == "all":
        error = np.zeros((curves.M,curves.N))
        status = np.empty(shape=(curves.M),dtype=np.str_)
        init = np.zeros((curves.M, model.ntheta))
        final = np.zeros((curves.M, model.ntheta))
        fit_sec = np.zeros(curves.M)
        tags = curves.tags
        for i, curve in enumerate(curves):
            start = time.time()
            try:
                _, status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
                    fit_id(curve,model,till).values()
            except KeyboardInterrupt:
                break
            except:
                status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
                    ("f", np.nan, np.nan, np.nan, time.time()-start)
                

    if isinstance(idx, list):
        error = np.zeros((len(idx),curves.N))
        status = np.empty(shape=(len(idx)),dtype=np.str_)
        init = np.zeros((len(idx), model.ntheta))
        final = np.zeros((len(idx), model.ntheta))
        fit_sec = np.zeros(len(idx))
        tags = list()
        for i,id_ in enumerate(idx):
            curve = curves[id_]
            try:
                res = fit_id(curve,model,till)
                tag, status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
                    res.values()
                tags.append(tag)
            except KeyboardInterrupt:
                break
            except:
                print("fit failed")

    return {"tags":tags,
            "status":status, "theta0":init, "theta":final,\
            "error":error, "time":fit_sec}

#def fit_idx(curves,model,till,idx):
#    """ 
#        curve   : 1xN shaped M full curves
#        model   : which model used to fit the curve 
#        till    : number of anchors from the  start to fit 
#        idx     : id of the curve
#    """
#    res = fit_id(curves[idx],model,till)
#    return res.values()
          
def fit_id(curve,model,till):
    """ 
        curves  : 1xN shaped M full curves
        model   : which model used to fit the curve 
        till    : number of anchors from the  start to fit 
    """
    start = time.time()
    model.fit(curve[:till])
    end = time.time()
    if model.status:
        status = "s"
    else:
        status = "f"
    return {"tag":curve.tag, "status":status,
            "theta0":model._theta0, "theta":model.theta, \
                    "error":SquaredError(model,curve), "time":end-start}

