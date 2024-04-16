"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : model/param.py
"""

import numpy as np
import sympy as sym
import math
from scipy.optimize import minimize, curve_fit, brute

class ParametricModelFactory():
    def __init__(self, config, n_param):

        self._ntheta = n_param

        self.config = config

        self.status = False

        self.xi = sym.IndexedBase('x')
        self.yi = sym.IndexedBase('y')
        self.ts = sym.symbols("t0:{}".format(self.ntheta))
        self.N, self.x = sym.symbols("N x")
        self.i = sym.symbols('i', integer=True)
        if not "init" in self.config.keys(): 
            self._theta0 = np.zeros(self.ntheta)
        elif config["init"] == "uniform":
            self._theta0 = np.random.uniform(size=self.ntheta)
        elif config["init"] == "uniform_wide":
            self._theta0 = np.random.uniform(0,5,size=self.ntheta)
        elif config["init"] == "normal":
            self._theta0 = np.random.normal(size=self.ntheta)
        elif config["init"] == "zeros":
            self._theta0 = np.zeros(self.ntheta)
        elif config["init"] == "ones":
            self._theta0 = np.ones(self.ntheta)
        else:
            self._theta0 = np.ones(self.ntheta)
        self.theta = self._theta0
        # These need to be defined in the child class only
        self._obj_expr = None 
        self._model_expr = None 
        self._theta_ranges = None # should be a nparam lengeht tuple of slices

    def _warm_start(self,curve):
        if self._theta_ranges == None:
           self._theta_ranges = [slice(0,1,0.25)]*self._ntheta
        self._lambdify__()
        def obj (param,model,curve):
            return np.mean((model(curve.anchors, *param)-curve.labels)**2)

        self._theta0 = brute(obj,self._theta_ranges,\
                        args=(self.model,curve),full_output=False, finish=None,\
                        Ns=1, workers=1)
        print(self._theta0)


    def _lambdify(self):
        assert (self._obj_expr is not None) or self._model_expr is not None
        if self._obj_expr is not None:
            if not "fit" in self.config.keys():
                self.obj = sym.lambdify((self.ts,self.xi,self.yi,self.N),
                    self._obj_expr)
            elif self.config["fit"] == "log":
                self.obj = sym.lambdify((self.ts,self.xi,self.yi,self.N),
                    sym.log(self._obj_expr+1))
            self.jac_obj = lambda theta, x, y, N:\
            [sym.lambdify((self.ts,self.xi,self.yi,self.N),
        sym.diff(sym.log(self._obj_expr+1),t))(theta,x,y,N) for t in self.ts]
        else:
            self.obj = None
        
        if self._model_expr is not None:
            if not "fit" in self.config.keys():
                self.model = sym.lambdify((self.x,*self.ts), self._model_expr)
            elif self.config["fit"] == "log":
                self.model = sym.lambdify((self.x,*self.ts),
                    sym.log(self._model_expr+1))

            self.jac_model = lambda x, *theta: \
                np.array([sym.lambdify((self.x,*self.ts),
                sym.diff(sym.log(self._model_expr+1),t))(x,*theta)\
                        for t in self.ts]).T
        else:
            self.model = None

    def _lambdify__(self):
        assert (self._obj_expr is not None) or self._model_expr is not None
        if self._obj_expr is not None:
            self.obj = sym.lambdify((self.ts,self.xi,self.yi,self.N),
                    self._obj_expr)
            self.jac_obj = lambda theta, x, y, N:\
            [sym.lambdify((self.ts,self.xi,self.yi,self.N),
            sym.diff(self._obj_expr,t))(theta,x,y,N) for t in self.ts]
        else:
            self.obj = None
        
        if self._model_expr is not None:
            self.model = sym.lambdify((self.x,*self.ts), self._model_expr)

            self.jac_model = lambda x, *theta: \
                np.array([sym.lambdify((self.x,*self.ts),
                sym.diff(self._model_expr,t))(x,*theta) for t in self.ts]).T
        else:
            self.model = None

    @classmethod
    def lambdify(cls,config):
        obj = cls(config)
        obj._lambdify__()
        return obj

    @property
    def model_expr(self):
        return self._model_expr

    @property
    def obj_expr(self):
        return self._obj_expr

    @property
    def theta(self):
        """
            Return the parameters of the model 
        """
        return self._theta

    @theta.setter
    def theta(self, values):
        """
            Set the parameters of the model 
            value   : 1xK array of parameters 
        """
        self._theta = values

    @property
    def ntheta(self):
        """
            Return the parameters of the model 
        """
        return self._ntheta

    @property
    def get_func(self):
        """
            Underlying function
        """
        return sym.lambdify((self.ts,self.x), self._model_expr)

    @property
    def get_jac_model(self):
        """
            Jacobian lambda function
        """
        return self.jac_model

    @property
    def get_jac_obj(self):
        """
            Jacobian Calculation
            xs  : 1xN anchors
        """
        return self.jac_obj

    def _fit(self,curve):
        """
            Fit to the 
            curve   : object returned from our database
        """
        if self.config["opt"] == "lm":
            assert (self.model is not None)
            if self.config["jac"]:
                results = curve_fit(self.model,curve.anchors,curve.labels,
                    p0=self._theta0,
                    jac=self.jac_model,full_output=True,maxfev=10000)
            else:
                results = curve_fit(self.model,curve.anchors,curve.labels,
                    p0=self._theta0,
                    jac=None,full_output=True,maxfev=10000)
            if results[-1] == 1 or results[-1] == 2 or results[-1] == 3 \
                    or results[-1] == 4:
                self.status = True
            self.theta = results[0]
        else:
            assert (self.obj is not None)
            results = minimize(self.obj,self._theta0, 
                    args=(curve.anchors,curve.labels,curve.N-1),
                    method=self.config["opt"],jac=self.jac_obj)
            self.theta = results.x
            self.status = results.success
        return self.status


    def fit(self,curve):
        """
            Fit to the 
            curve   : object returned from our database
        """
        self._lambdify()
        ### If we choose the warm initialization the possibiilty that we don't 
        ### have memory available is a possibility. If we exceeed the memory
        ### we initialize the parameters with ones.
        if self.config["init"] == "warm":
            try:
                self._warm_start(curve)
            except:
                self._theta0 = np.ones(self.ntheta)
        if not "fit" in self.config.keys():
            return self._fit(curve)
        if self.config['fit'] == "log":
            return self._fit(curve.log_transform())
        

    def predict(self,xs):
        """
            Pred with the model at
            xs  : 1xN anchors
        """
        #return self.model(xs,*self.theta)
        self._lambdify__()
        return self.model(xs,*self.theta)


class EXP2(ParametricModelFactory):
    def __init__(self,config):
        
        # How many parameters does the model have 
        nparam = 2

        super().__init__(config, nparam)
        
        # Objective function with the indexing
        self._obj_expr = (sym.Sum(
                (self.ts[0]*sym.exp(-self.ts[1]*self.xi[self.i])-
                    self.yi[self.i])**2,
                (self.i,0,self.N)))

        # Function for the model
        self._model_expr = self.ts[0]*sym.exp(-self.ts[1]*self.x)


class BNSL(ParametricModelFactory):
    def __init__(self,config):
        
        # How many parameters does the model have 
        if config.get("nbreak"):
            self.nbreak = config["nbreak"]
        else:
            self.nbreak = 0

        nparam = 3+self.nbreak*3
        super().__init__(config, nparam)
        
        # Function for the model
        _model_expr_base = self.ts[0]+self.ts[1]*self.x**-self.ts[2]
        if self.nbreak>0:
            _model_expr_n = [(1.+(self.x/self.ts[3*i])**(1./self.ts[3*i+1]))\
                    **(-self.ts[3*i+2]*self.ts[3*i+1])\
                    for i in range(1,self.nbreak+1)]
        else:
            _model_expr_n = [1]
        self._model_expr = (_model_expr_base * math.prod(_model_expr_n))


