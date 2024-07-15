"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : main.py
"""
# Import external 
from sacred import Experiment 
import xarray as xr
import polars as ps
import numpy as np
import sys
import inspect as insp
import importlib as implib
import matplotlib.pyplot as plt
import os
# Import local
from model import *
from data import *
from fit import *
from results import *
experiment_name = 'name_your_experiment'
ex = Experiment(experiment_name,save_git_info=False)

@ex.capture
def get_run_dir(_run):
    if _run._id is not None:
        return os.path.join(_run.experiment_info["name"],_run._id)
    else:
        return _run.experiment_info["name"]

@ex.capture
def get_experiment_dir(_run):
    return _run.experiment_info["name"]


@ex.capture
def get_observer_dir(_run):
       return _run.observers[0].basedir


@ex.config
def my_config():
    """ 
        Configuration goes here everything is arranged as a nested dictionary.
        Seed is generated randomly and stored in the config.json
        when you add a file observer. 
    """
    conf = dict()

    #conf["database"] = {
    #        "type":"xarray_dataset"
    #        "name":"LCDB.nc",   
    #        "kwargs":None       # this should include split/normalize at least
    #        }
    conf["database"] = {
            "type":"csv",
            "filename":"classification.csv",   
            "kwargs":None       # this should include split/normalize at least
            }

    conf["model"] = {
            "name":"EXP2",      # model name this allows selection of the model
            "init":"ones",    # how to initiliaze the model parameters
            # "fit":"log",        # fit with log transformation
            "opt": "L-BFGS-B",  # all the options in scipy.optimize.minimize "jac": True,        # exact jacobian usage instead of finite diff.
            }
    # conf["model"] = {
    #         "name":"BNSL",      # model name this allows selection of the model
    #         "nbreak":0,         # model name this allows selection of the model
    #         "init":"warm",      # how to initiliaze the model parameters
    #         "fit":"log",        # fit with log transformation
    #         "opt":"lm",        # all the options in scipy.optimize.minimize
    #                             # and lm for scipy.optimize.curve_fit
    #         "jac": False,        # exact jacobian usage instead of finite diff.
    #         }

    # Add experiment functions here.
    conf["exp"] = {
            "fit":fit,
            } 

    # Add experiment configuration
    conf["fit"] = {
            "which":[0],
            "till":50,
            "restarts":None,
            }

    # Get the available models
    conf["models"] = dict()
    for a, b in insp.getmembers(implib.import_module("model"),insp.isclass):
        conf["models"].setdefault(a, [ ]).append(b) 

    # Save related configuration
    conf["save"] = {
            "type":"brief",      # or detail
            "name":"last.csv",   # name of the file
            }
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]  # Blue, Yellow, Red
positions = [0, 0.5, 1]  # Positions of the colors along the colormap
cmap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)))
@ex.automain
def run(seed,conf,_run):
    """ 
        Main experiment will run in this one
        seed    : seed of the run (automatically determined by sacred)
        conf    : configuration that we specify
    """
    curves = Database(conf["database"]).get_curves()
    curve = curves[0]
    curve.anchors = np.arange(1,100)
    curve.labels = np.exp(-curve.anchors)+np.random.normal(0.,0.001,size=(len(curve)))
    #curve.labels *=100

    model = conf["models"][conf["model"]["name"]][0](conf["model"])
    model._lambdify__()
    print(model._model_expr)
    #model.obj((1,1),curve.anchors,curve.labels,len(curve)-1)
    t0 = np.linspace(-10,10,100)
    t1 = np.linspace(0,1,100)
    t0_, t1_ = np.meshgrid(t0,t1)
    res = np.zeros_like(t1_)

    for i in range(t1_.shape[0]):
        for j in range(t1_.shape[1]):
            res[i, j] =  model.obj((t0_[i,j],t1_[i,j]),curve.anchors,curve.labels,len(curve)-1)   
    idx = np.unravel_index(np.nanargmin(res),res.shape)
    print(t0_[idx],t1_[idx],res[idx])

    X = t0_; Y = t1_
    plt.imshow(res, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], origin='lower', cmap=cmap, aspect='auto',vmin=0., vmax=np.min(res)*1e5)
    plt.xlabel("t0")
    plt.ylabel("t1")
    #plt.imshow(res, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], cmap='magma', aspect='auto')
    plt.colorbar()  # Add a color bar to indicate values
    plt.show()

