"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : data/database.py
    #NOTE   : Maybe it is a waste of time, but oke
"""
import numpy as np
import polars as ps
class Curve():
    """
        This class holds one learning curve that you want to fit in tuples
        with their corresponding anchor
        iterator for the curves.
    """

    def __init__(self,anchors,labels,tag=None):
        """
           anchor   : 1xN array for anchor points of the learning curves
           labels   : 1xN array labels of the learning curves
           tags     : unique id of the curve
        """
        if tag is not None:
            self.tag = tag
        else:
            self.tag = "none"
        self.N = labels.shape[0] # This is the number of anchors needed
        assert len(anchors.shape) == 1 and len(labels.shape) == 1
        self.anchors = anchors
        self.labels = labels

    def __len__(self):
        return (self.N)

    def __str__(self):
        return ("anchors:{}-labels:{}".
                format(self.anchors,self.labels))

    def shape(self):
        return self.__len__()

    def __getitem__(self,*args): 
        return Curve(self.anchors[args[0]],self.labels[args[0]])

    def clip(self):
        """
            This method will clip all the nan values to get your data ready for 
            fitting.
        """
        idxs = np.where(np.isnan(self.labels))
        for idx in idxs:
            if idx.size != 0:
                self.anchors = self.anchors[:,:idx[0]]
                self.lables = self.labels[:,:idx[0]]
    
    def __iter__(self):
        self.counter = -1
        return self

    def __next__(self):
        if self.counter == self.N-1:
            raise StopIteration
        self.counter += 1
        return (self.anchors[self.counter], self.labels[self.counter])

    def normalize(self):
        # Need to fill this with some variants that we might discuss 
        pass
    def log_transform(self):
        self.labels = np.log(self.labels+1)
        return Curve(self.anchors, self.labels)


class Curves():
    """
        This class holds all the learning curves that you want to fit in tuples
        with their corresponding anchors in a clean format with a built in
        iterator for the curves.
    """

    def __init__(self,anchors,labels,tags=None):
        """
           anchor   : 1xN or MxN array for anchor points of the learning curves
           labels   : MxN array labels of the learning curves
           tags     : 1xM array of unique ids
        """
        self.M = labels.shape[0] # This is the number of curves we have 
        self.N = labels.shape[1] # This is the number of anchors needed
        if tags is not None:
            self.tags = tags
        else:
            self.tags = ["none"] * self.M
        # if you provide one dimensional anchor dataa for all the curves 
        # same anchors will be used. 
        if len(anchors.shape) == 1:
            assert self.N == anchors.shape[0]
            self.anchors = np.tile(anchors,(self.M,1))
        else:
            assert self.N == anchors.shape[1]
            self.anchors = anchors

        self.labels = labels
        self.clip()

    def __len__(self):
        return ((self.anchors.shape,self.labels.shape))

    def __str__(self):
        return ("anchors_shape:{}-labels_shape:{}".
                format(self.anchors.shape,self.labels.shape))

    def shape(self):
        return self.__len__()

    def __getitem__(self,*args): 
        return Curve(self.anchors[args[0]],self.labels[args[0]],self.tags[args[0]])

    def clip(self):
        """
            This method will clip all the nan values to get your data ready for 
            fitting.
        """
        idxs = np.where(np.isnan(self.labels))
        for idx in idxs:
            if idx.size != 0:
                self.anchors = self.anchors[:,:idx[0]]
                self.lables = self.labels[:,:idx[0]]
    
    def __iter__(self):
        self.counter = -1
        return self

    def __next__(self):
        if self.counter == self.M-1:
            raise StopIteration
        self.counter += 1
        return Curve(self.anchors[self.counter], self.labels[self.counter])

    def normalize(self):
        pass

class Database():
    def __init__(self, conf):
        if conf["type"] == "csv":
            assert conf.get("filename")
            self.data = ps.read_csv(conf["filename"])
            self.N = self.data["N"].to_numpy()
            self.curves = self.data.drop("N")
            self.idx = self.curves.columns
            self.curves = self.curves.to_numpy()
        else:
            NotImplementedError 
    def get_curves(self,idx=None):
        if idx==None:
            return Curves(self.N,self.curves.T,self.idx)
        else:
            return Curves(self.N,(self.curves.T)[idx],
                        np.array(self.idx)[idx].tolist())
    def get_learner(self,idx):
        pass 
    def get_dataset(self,idx):
        pass 
    def get_seed(self,idx):
        pass
    def get_train(self,idx):
        pass
    def get_valid(self,idx):
        pass
    def get_test(self,idx):
        pass

    # Generator for the curves in the database

