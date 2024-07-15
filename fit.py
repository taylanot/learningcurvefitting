"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : fit.py
"""
import time 
import numpy as np
import warnings 
#warnings.filterwarnings("ignore",category=RuntimeWarning)
def SquaredError(model, curve):
        return (model.predict(curve.anchors)-curve.labels)**2

def fit(curves,model,config):

    """ 
        Fit all the data
        curves  : MxN M full curves with N acnhors
        model   : which model used to fit the curve
        till    : number of anchors from the  start to fit
    """
    def fit_all(fit_func,config):
        error = np.zeros((curves.M,curves.N))
        status = np.empty(shape=(curves.M),dtype=np.str_)
        init = np.zeros((curves.M, model.ntheta))
        final = np.zeros((curves.M, model.ntheta))
        fit_sec = np.zeros(curves.M)
        tags = curves.tags
        for i, curve in enumerate(curves):
            start = time.time()
            try:
                _, status[i], init[i,:], final[i,:], error[i,:],\
                        fit_sec[i] = fit_func(curve,model,config).values()
            except KeyboardInterrupt:
                break
            except Exception as e:
                warnings.warn(str(e))
                status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
                    ("f", np.nan, np.nan, np.nan, time.time()-start)
        return {"tags":tags,
                "status":status, "theta0":init, "theta":final,\
                "error":error, "time":fit_sec}
               

    def fit_ids(fit_func,config):
        idx = config["which"]
        if isinstance(idx, list):
            error = np.zeros((len(idx),curves.N))
            status = np.empty(shape=(len(idx)),dtype=np.str_)
            init = np.zeros((len(idx), model.ntheta))
            final = np.zeros((len(idx), model.ntheta))
            fit_sec = np.zeros(len(idx))
            tags = list()
            for i,id_ in enumerate(idx):
                curve = curves[id_]
                start = time.time()
                try:
                    res = fit_func(curve,model,config)
                    _, status[i], init[i,:], final[i,:], error[i,:],\
                            fit_sec[i] = res.values()
                    tags.append(curves.tags[id_])
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    warnings.warn(str(e))
                    status[i], init[i,:], final[i,:], error[i,:], fit_sec[i] = \
                        ("f", np.nan, np.nan, np.nan, time.time()-start)
                    tags.append(curves.tags[id_])
        return {"tags":tags,
                "status":status, "theta0":init, "theta":final,\
                "error":error, "time":fit_sec}
          
    if config["restarts"] is not None:
        fit_func = fit_id_restart
    else:
        fit_func = fit_id

    if (isinstance(config["which"],str) and config["which"]=="all"):
        return fit_all(fit_func,config)
    elif (isinstance(config["which"],int)):
        return fit_id(curves[config["which"]],model,config)
    elif (isinstance(config["which"],list)):
        return fit_ids(fit_func,config)
    else:
        NotImplementedError

def fit_id(curve,model,config):
    """ 
        curves  : 1xN shaped M full curves
        model   : which model used to fit the curve 
        till    : number of anchors from the  start to fit 
    """
    start = time.time()
    model.fit(curve[:config["till"]])
    end = time.time()
    if model.status:
        status = "s"
    else:
        status = "f"
    return {"tag":curve.tag, "status":status,
            "theta0":model._theta0, "theta":model.theta, \
                    "error":SquaredError(model,curve), "time":end-start}

###############################################################################
# I think I can add validation as an option here 
###############################################################################
def fit_id_restart(curve,model,config):
    """ 
        curves  : 1xN shaped M full curves
        model   : which model used to fit the curve 
        till    : number of anchors from the  start to fit 
    """

    assert model._theta_ranges is not None, "model_theta_ranges does not exist"

    start = time.time()
    error = np.zeros((config["restarts"]))
    status = np.empty(shape=(config["restarts"]),dtype=np.str_)
    theta0 = np.zeros((config["restarts"],model._ntheta))
    theta = np.zeros((config["restarts"],model._ntheta))
    times = np.zeros(config["restarts"])

    for i in range(config["restarts"]):
        start2 = time.time()
        model() # this is for resampling the initial 
        try:
            _, status[i],theta0[i,:],theta[i,:],errors,times[i] = \
                fit_id(curve,model,config).values()
            error[i] = np.mean(errors)
        except KeyboardInterrupt:
            break
        except Exception as e:
            warnings.warn(str(e))
            status[i],theta0[i,:],theta[i,:],error[i],times[i] = \
                    ("f", np.nan, np.nan, np.inf, time.time()-start2)
    idx = np.argmin(error)
    model.theta = theta[idx,:]
    model._theta0 = theta0[idx,:]
    end = time.time()

    return {"tag":curve.tag, "status":status[idx],
            "theta0":theta0[idx,:], "theta":theta[idx,:], \
                    "error":SquaredError(model,curve), "time":end-start}
