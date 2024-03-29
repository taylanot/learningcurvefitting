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
from model import *
from data import *
from fit import *

def num_jac(f,x,args=None,method='central',h=0.01, ):
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
        return (f(x_,*args) - f(_x,*args))/(2*h)
    elif method == 'forward':
        return (f(x_,*args) - f(x,*args))/h 
    elif method == 'backward':
        return (f(x,*args) - f(_x,*args))/h 

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

def param_test_factory(model,x = np.arange(1,20),tol=1e-3):
    theta = model._theta0
    model.theta = theta
    y = model.predict(x)
    method = "central"
    curve = Curve(x,y)
    h = 1e-6

    class Test(unittest.TestCase):
        def test_obj_jacobian(self):
            args = (x,y,len(x)-1)
            jac_ = num_jac(model.obj,theta,args,method,h)
            jac = model.get_jac_obj(theta,x,y,len(x)-1)
            self.assertTrue(np.allclose(jac,jac_, tol, tol))
        def test_model_jacobian(self):
            for xi in x:
                jac_ = num_jac(model.get_func,theta,[xi],method,h)
                jac = model.get_jac_model(xi,*theta)
                self.assertTrue(np.allclose(jac,jac_, tol, tol))
        def test_fit(self):
            model._theta0[:] = 0
            model.fit(curve)
            self.assertTrue(np.allclose(model.theta,np.ones(model.ntheta)
                ,tol,tol))
    return Test

models = dict()
for a, b in insp.getmembers(implib.import_module("model"),insp.isclass):
        models.setdefault(a, [ ]).append(b) 

for name,uninit_model in models.items():
    if not (name == "LAST1" or name == "LASTGRAD"):
        config = { "name":name,      
                    "init":"ones",    
                    "opt": "lm",  
                    "jac": True }
        model = uninit_model[0](config)
        class Tests(param_test_factory(model)):
            pass

if __name__ == '__main__':
    unittest.main()


