"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : model/__init__.py
"""
from .param import (BNSL,EXP2,POW2,LOG2,LIN2,ILOG2,POW3,EXP3,VAP3,EXPP3,LOGPOW3,
                    POW4,WBL4,EXP4)
from .heuristic import (LAST1,LASTGRAD)

__all__ = [
    "LAST1",
    "LASTGRAD",
    "EXP2",
    "POW2",
    "LOG2",
    "LIN2",
    "ILOG2",
    "POW3",
    "EXP3",
    "VAP3",
    "EXPP3",
    "LOGPOW3",
    "POW4",
    "WBL4",
    "EXP4",
    "BNSL",
]
