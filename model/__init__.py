"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : model/__init__.py
"""
from .param import (BNSL)
from .heuristic import (LAST1,LASTGRAD)

__all__ = [
    "LAST1",
    "LASTGRAD",
    #"EXP2",
    "BNSL",
]
