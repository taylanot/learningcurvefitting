"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : main.py
"""
# Import external 
from sacred import Experiment 
import xarray as xr
import numpy as np
import sys
import inspect as insp
import importlib as implib
import os
# Import local
from model import *
from data import *
from fit import *

experiment_name = 'name_your_experiment'
ex = Experiment(experiment_name,save_git_info=False)

@ex.capture
def get_experiment_dir(_run):
    if _run._id is not None:
        return os.path.join(_run.experiment_info["name"],_run._id)
    else:
        return _run.experiment_info["name"]

@ex.config
def my_config():
    """ 
        Configuration goes here everything is arranged as a nested dictionary.
        Seed is generated randomly and stored in the config.json
        when you add a file observer. 
    """
    conf = dict()

    conf["database"] = {
            "name":"LCDB.nc",   
            "kwargs":None       # this should include split/normalize at least
            }

    conf["model"] = {
            "name":"EXP2",      # model name this allows selection of the model
            "init":"normal",    # how to initiliaze the model parameters
            "opt": "L-BFGS-B",  # all the options in scipy.optimize.minimize
                                # and lm for scipy.optimize.curve_fit
            "jac": True,        # exact jacobian usage instead of finite diff.
            }

    conf["run"] = {
            "which":"all",
            "at":10
            }

    # Add experiment functions here.
    conf["exps"] = {
            "fit":fit,
            "fit_id":fit_id,
            } 

    conf["models"] = dict()
    for a, b in insp.getmembers(implib.import_module("model"),insp.isclass):
        conf["models"].setdefault(a, [ ]).append(b) 

@ex.automain
def run(seed,conf,_run):
    """ 
        Main experiment will run in this one
        seed    : seed of the run (automatically determined by sacred
        conf    : configuration that we specify
    """
    save_dir = get_experiment_dir()
    # Get the database to be used

    # It can be good to write a database class fro wrapping this nicely for any
    # other database, but for now I think it is a waste of time...
    curves = Curves(np.arange(20),np.array([np.exp(-np.arange(20)),
                                            np.exp(-2*np.arange(20))]))
    #database = xr.load_dataset(conf["database"]["name"])
    #print(database)

    #for curve in database[{"inner_seed":0,"outer_seed":0,"ttv":0}].curves:
    # This is how tyou access one "curve"
    #print(database[{"dataset":0,"learner":0,"inner_seed":0,"outer_seed":0,"ttv":0}].values)
    #curve = database[{"dataset":0,"learner":0,"inner_seed":0,"outer_seed":0}].curves
    # Get the model
    model = conf["models"][conf["model"]["name"]][0](conf["model"])

    # Select the experiment
    if (isinstance(conf["run"]["which"],str) and conf["run"]["which"]=="all"):
        result = conf["exps"]["fit"](
            curves,model,
            conf["run"]["at"]
            )

    elif (isinstance(conf["run"]["which"],int)):
        result = conf["exps"]["fit_id"](
            curves[conf["run"]["which"]],
            model,conf["run"]["at"]
            )
    else:
        NotImplementedError
    print(result)

