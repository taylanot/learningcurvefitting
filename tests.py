"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : tests.py

    Here all unittests will live!
"""
# Import external 
import numpy as np
import inspect as insp
import importlib as implib
#
import unittest 
# Import local
from model.param import *
from data import *
from fit import *

def num_jac(f,x,method='central',h=0.01, *args):
    """
        Evaluate Jacobian of a function at a given point.
    """
    if isinstance(x,list):
        x = np.array(x)
    assert x.ndim ==1, "x should be one dimensional"

    dx = len(x)

    x_ = np.tile(x,(dx,1)) + np.eye(dx)*h
    _x = np.tile(x,(dx,1)) - np.eye(dx)*h
    
    if method == 'central':
       return (f(x_,args) - f(_x,args))/(2*h)

    elif method == 'forward':
        return (f(x_,args) - f(x,args))/h 

    elif method == 'backward':
        return (f(x,args) - f(_x,args))/h 

    else:
        raise NotImplementedError

"""
    Oke the test class for the ParametricModels will be checking the jacobians 
    with the numerical counterpart and checking the optimization. To do that
    you have to find and initialize all the available models from param.py,
    then generate data from its own predict function with _theta0 = 1 and try
    to fit to that data with another initialization this might be extra. 
    Creating a test factory is a great idea for this i think. Need to adjust
    above num_jac for the reversed args intake.
"""

def param_test_factory(model):
    class Test(unittest.TestCase):
        def test_obj_jacobian(self):
            self.assertEqual(model, 1)
        def test_model_jacobian(self):
            self.assertEqual(model, 1)
        def test_fit(self):
            self.assertEqual(model, 1)

    return Test

class TEST_EXP2(param_test_factory(1)): 
    pass

## I have to rethink this maybe it is better to write a test suite here.    
#class Test_parametric_models(unittest.TestCase):
#    def __init__(self, *args, **kwargs):
#        super(Test_parametric_models, self).__init__(*args, **kwargs)
#        x = np.arange(10); y = np.exp(x)
#        self.tol = 1e-3
#        self.curve = Curve(x,y)
#        self.conf = dict()
#        
#    def test_Last1(self):
#        self.conf["model"] = {
#                    "name":"Last1",
#                    "kwargs":None, # this should include initialization at least
#                    }
#        model = Last1(self.conf["model"])
#        model.fit(self.curve)
#        self.assertEqual(model._theta,np.exp(9))
#
#    def gradients(self):
#        self.conf["models"] = dict()
#        for a, b in insp.getmembers(implib.import_module("model"),insp.isclass):
#            self.conf["models"].setdefault(a, [ ]).append(b) 
#
#        model_names = self.conf["models"].keys()
#        for model_name in model_names:
#            if model_name != "Last1":
#                self.conf = {
#                            "name":model_name,
#                            "kwargs":None, # this should include initialization at least
#                            }
#                self.model = self.conf["models"](self.conf)
#
#    def test_gradients(self):
#            self.assertAlmostEqual(model.exact_grad(self.x),
#                        model.approx_grad(self.x),
#                        None,
#                        self.tol)
#



if __name__ == '__main__':
    unittest.main()


