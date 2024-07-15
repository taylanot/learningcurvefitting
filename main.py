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
#import matplotlib.pyplot as plt
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

    #conf["model"] = {
    #        "name":"EXP2",      # model name this allows selection of the model
    #        "init":"ones",    # how to initiliaze the model parameters
    #        #"fit":"log",        # fit with log transformation
    #        "opt": "L-BFGS-B",  # all the options in scipy.optimize.minimize "jac": True,        # exact jacobian usage instead of finite diff.
    #        }
    conf["model"] = {
            "name":"BNSL",      # model name this allows selection of the model
            "nbreak":0,         # model name this allows selection of the model
            "init":"warm",      # how to initiliaze the model parameters
            "fit":"log",        # fit with log transformation
            "opt":"lm",        # all the options in scipy.optimize.minimize
                                # and lm for scipy.optimize.curve_fit
            "jac": False,        # exact jacobian usage instead of finite diff.
            }

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

@ex.automain
def run(seed,conf,_run):
    """ 
        Main experiment will run in this one
        seed    : seed of the run (automatically determined by sacred)
        conf    : configuration that we specify
    """
    # Get the database to be used

    # It can be good to write a database class fro wrapping this nicely for any
    # other database, but for now I think it is a waste of time...
    # exponential
    #curves = Curves(np.arange(20),np.array([np.exp(-np.arange(20)),
    #                                        np.exp(-2*np.arange(20))]))

    curves = Database(conf["database"]).get_curves()
    #print(curves)
    #database = xr.load_dataset(conf["database"]["name"])
    #print(database)

    #for curve in database[{"inner_seed":0,"outer_seed":0,"ttv":0}].curves:
    # This is how tyou access one "curve"
    #print(database[{"dataset":0,"learner":0,"inner_seed":0,"outer_seed":0,"ttv":0}].values)
    #curve = database[{"dataset":0,"learner":0,"inner_seed":0,"outer_seed":0}].curves
    # Get the model
    model = conf["models"][conf["model"]["name"]][0](conf["model"])
    #x = np.arange(1,20)
    #curves = Curves(x,np.array([2+x.astype(float)**(-2),
    #                            3*x.astype(float)**(-2),
    #                            1+x.astype(float)**(-2)]))
    #curves = Curve(np.arange(1,20),np.array(model.predict(np.arange(1,20))))

    # Select the experiment
    result = conf["exp"]["fit"](curves,model,conf["fit"])
    #if (isinstance(conf["fit"]["which"],str) and conf["fit"]["which"]=="all"):
    #    result = conf["with"]["fit"](
    #        curves,model,conf["fit"]["till"],conf["fit"]["which"]
    #        )

    #elif (isinstance(conf["fit"]["which"],int)):
    #    result = conf["with"]["fit_id"](
    #        curves[conf["fit"]["which"]],model,conf["fit"]["till"]
    #        )
    #elif (isinstance(conf["fit"]["which"],list)):
    #    result = conf["with"]["fit"](
    #        curves,model,conf["fit"]["till"],conf["fit"]["which"]
    #        )

    #else:
    #    NotImplementedError

    exp_path=get_experiment_dir(),
    if hasattr(_run,"observers") and len(_run.observers)!=0:
        obs_path=get_observer_dir(),
    else:
        obs_path = "./"
    path = os.path.join(obs_path[0],exp_path[0],str(conf['fit']['till']),
            os.path.splitext(conf["database"]["filename"])[0])
    mlcxx_type(result,conf["save"],path)
    #ex.add_artifact(config['save']["name"])
