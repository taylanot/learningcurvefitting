"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : results.py
"""
import polars as ps 
import numpy as np
import os
import pathlib

def mlcxx_type(result, conf,path=None):
    if path is None:
        path = "."
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
    if conf["type"] == "brief":
        if len(result["error"].shape) != 1:
            df = ps.DataFrame(result["error"][:,-1][np.newaxis,:],orient='row')
            df.columns = result["tags"]
        else:
            df = ps.DataFrame(result["error"],orient='row')
            df.columns = [result["tag"]]
        df.write_csv(os.path.join(path,conf["name"]),separator=",")
    else:
        NotImplementedError
