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
# Import local
from model import *
from fit import *

ex = Experiment('my_experiment',save_git_info=False)

@ex.config
def my_config():
    """ 
        Configuration goes here everything is arranged as a nested dictionary.
        Seed is generated randomly and stored in the config.json when you add a file observer.
    """
    conf = dict()

    conf["database"] = {
            "name":None,
            "kwargs":None # this should include split and normalize at least
            }

    conf["model"] = {
            "name":"Last1",
            "kwargs":None, # this should include initialization at least
            }

    conf["run"] = {
            "which":"all",
            "at":10,
            "tos":[-1,-2,-3]
            }

    # Add experiment functions here.
    conf["exps"] = {
            "fit":fit,
            "fit_id":fit_id,
            } 

    conf["parametric_models"] = dict()
    for a, b in insp.getmembers(implib.import_module("model"),insp.isclass):
        conf["parametric_models"].setdefault(a, [ ]).append(b) 
        


@ex.capture
def echo(what):
    """ 
        This is a simple function to capture.
    """
    print(what)

@ex.capture
def fit(curves,model,at,tos):
    """ 
        Fit all the data
        curves  : MxN M full curves with N acnhors
        model   : which model used to fit the curve
        at      : number of anchors from the  start to fit
        tos     : which anchors to test the extrapolation 
    """
    for curve in curves:
       fit_id(curve,model,at,tos) 
     
@ex.capture
def fit_id(curve,model,at,tos):
    """ 
        curve   : 1xN M full curves with N acnhors
        model   : which model used to fit the curve 
        at      : number of anchors from the  start to fit 
        tos     : which anchors to test the extrapolation 
    """
    model.fit(curve[:at])
    for to in tos:
        model.error(curve[to])
    return 

@ex.automain
def run(seed,conf):
    """ 
        Main experiment will run in this one
        seed    : seed of the run (automatically determined by sacred
        conf    : configuration that we specify
    """
    np.random.seed(seed)            # Set the seed for numpy
    # Get the database to be used
    if conf["database"]["name"] is None:     
        database = np.zeros((10,100))
    else:
        NotImplemented("Not present yet...")
    # Get the model
    model = Last1(conf["model"])
    
    # Select the experiment
    if (isinstance(conf["run"]["which"],str) and conf["run"]["which"]=="all"):
        conf["exps"]["fit"](
            database,model,
            conf["run"]["at"],conf["run"]["tos"]
            )

    if (isinstance(conf["run"]["which"],int)):
        conf["exps"]["fit_id"](
            database[conf["run"]["which"]],
            model,conf["run"]["at"],conf["run"]["tos"]
            )

    #if conf["fit"] == "all":
    #    fit(

    #print(xr.load_dataset(conf["dataset"]))
    #print(xr.backends.list_engines())
    #print(conf["parametric_models"])
    #model = Linear
    #conf["exps"][conf["what"]]("data")

#if __name__ == '__main__':
#    ex.run()
