"""
    @project: Learning Curve Extrapolation Example
    @author : Ozgur Taylan Turan, 2024
    @file   : example.py
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

    conf["model"] = {
            "name":"EXP2",      # model name this allows selection of the model
            "init":"zeros",    # how to initiliaze the model parameters
            "opt": "lm",  # all the options in scipy.optimize.minimize 
            "jac": False,        # exact jacobian usage instead of finite diff.
            }
    

    # Add experiment functions here.
    conf["exp"] = {"fit":fit} 

    # Add experiment configuration
    conf["fit"] = {
            "which":"all",
            "till":50,
            "restarts":10,
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

    ### Get the database to be used, for simplicity I am using this dataset,
    #### where we have equally distributed curves for e^-x e^-2x.
    curves = Curves(np.arange(20),np.array([np.exp(-np.arange(20)),
                                            np.exp(-2*np.arange(20))]))

    
    model = conf["models"][conf["model"]["name"]][0](conf["model"])

    ### Select the experiment, I am calling the fit function that is written
    result = conf["exp"]["fit"](curves,model,conf["fit"])
    

    ### You can select the you do not need to touch this area it is just 
    ### handeling the directories
    exp_path=get_experiment_dir(),
    if hasattr(_run,"observers") and len(_run.observers)!=0:
        obs_path=get_observer_dir(),
    else:
        obs_path = "./"
    path = os.path.join(obs_path[0],exp_path[0],str(conf['fit']['till']))

    ### This is my output function you have to write the output function yourself
    print_result(result,conf["save"],path)
    ### You can uncomment the following line to add artifects to your experiment
    # ex.add_artifact(conf['save']["name"])
