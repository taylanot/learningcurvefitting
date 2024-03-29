"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : model/param.py
"""

import numpy as np
import sympy as sym
from scipy.optimize import minimize, curve_fit

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
        elif config["init"] == "normal":
            self._theta0 = np.random.normal(size=self.ntheta)
        elif config["init"] == "zeros":
            self._theta0 = np.random.zeros(size=self.ntheta)
        elif config["init"] == "ones":
            self._theta0 = np.random.ones(size=self.ntheta)
        else:
            NotImplementedError

    def _lambdify(self):
        self.obj = sym.lambdify((self.ts,self.xi,self.yi,self.N),self._obj_expr)

        self.jac_obj = lambda theta, x, y, N:\
        [sym.lambdify((self.ts,self.xi,self.yi,self.N),
        sym.diff(self._obj_expr,t))(theta,x,y,N) for t in self.ts]

        self.model = sym.lambdify((self.x,*self.ts), self._model_expr)

        self.jac_model = lambda x, *theta: \
            np.array([sym.lambdify((self.x,*self.ts),
            sym.diff(self._model_expr,t))(x,*theta) for t in self.ts]).T

    @property
    def model_expr(self):
        return self._model_expr

    @model_expr.setter
    def model_expr(self, expr):
        return self._model_expr

    @property
    def obj_expr(self):
        return self._obj_expr

    @obj_expr.setter
    def obj_expr(self, expr):
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
    def get_jac_model(self, x, theta):
        """
            Jacobian lambda function
        """
        return self.jac_model(x,theta)

    @property
    def get_jac_obj(self,theta,x,y,N):
        """
            Jacobian Calculation
            xs  : 1xN anchors
        """
        return self.jac_obj(theta,x)

    def fit(self,curve):
        """
            Fit to the 
            curve   : object returned from our database
        """
        if self.config["opt"] == "lm":
            if self.config["jac"]:
                results = curve_fit(self.model,curve.anchors,curve.labels,
                    p0=self._theta0,
                    jac=self.jac_model,full_output=True)
            else:
                results = curve_fit(self.model,curve.anchors,curve.labels,
                    p0=self._theta0,
                    jac=None,full_output=True)
            if results[-1] == 1 or results[-1] == 3:
                self.status = True
            self.theta = results[0]
        else:
            results = minimize(self.obj,self._theta0, 
                    args=(curve.anchors,curve.labels,curve.N-1),
                    method=self.config["opt"],jac=self.jac_obj)
            self.theta = results.x
            self.status = results.success
        return self.status
        

    def predict(self,xs):
        """
            Predict with the model at
            xs  : 1xN anchors
        """
        return self.model(xs,*self.theta)


class EXP2(ParametricModelFactory):
    def __init__(self,config):

        nparam = 2

        super().__init__(config, nparam)

        obj = (sym.Sum(
                (self.ts[0]*sym.exp(-self.ts[1]*self.xi[self.i])-
                    self.yi[self.i])**2,
                (self.i,0,self.N)))

        model = self.ts[0]*sym.exp(-self.ts[1]*self.x)

        self._obj_expr = obj
        self._model_expr = model

        self._lambdify()
