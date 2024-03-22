"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : data/database.py
    #NOTE   : Maybe it is a waste of time, but oke
"""
import numpy as np

class Curves():
    """
        This class holds all the learning curves that you want to fit in tuples
        with their corresponding anchors in a clean format with a built in
        iterator for the curves.
    """

    def __init__(self,anchors,labels):
        """
           anchor   : 1xN or MxN array for anchor points of the learning curves
           labels   : MxN array labels of the learning curves
        """
        self.M = labels.shape[0] # This is the number of curves we have 
        self.N = labels.shape[1] # This is the number of anchors needed
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
        return (self.anchors[args[0]],self.labels[args[0]])

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
        return (self.anchors[self.counter], self.labels[self.counter])

    def normalize(self):
        pass

       

class Database():
    def __init__(self, conf):
        self.db = xr.load_dataset(conf["name"])
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

