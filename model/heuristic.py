import polars as ps
import numpy as np
from sklearn.metrics import pairwise_distances
"""
    @project: Learning Curve Extrapolation
    @author : Ozgur Taylan Turan, 2024
    @file   : model/heuristic.py
"""
class LAST1():
    def __init__(self,config):
        self.config = config
        self.ntheta = 1
    def fit(self,curve):
        self.theta = curve.labels[-1]
        self._theta0 = self.theta
        self.status = "s"
    def predict(self,xs):
        return self.theta

class LASTGRAD():
    def __init__(self,config):
        self.config = config
    def fit(self,curve):
        self.theta = (curve.labels[-1]-curve.labels[-2]) / \
                (curve.anchors[-1]-curve.anchors[-2])
        self.status = "s"
        self._theta0 = self.theta
    def predict(self,xs):
        return self.theta*xs

class MDS():
    def __init__(self,config):
        self.config = config
        self.ntheta = config["k"]
        self.database = ps.read_csv(config["database"],has_header=True).\
                                                            to_numpy()[:,1:]
    def cossim(self,vectors, vector):
        """
        Compute cosine similarity between one vector and multiple vectors.

        Args:
        - vector: A 1D array (reference vector).
        - vectors: A 2D array where each row is a vector to compare against.

        Returns:
        - similarities: A 1D array of cosine similarity values.
        """
        # Ensure input vector is 1D
        vector = np.ravel(vector)

        # Compute norms for the vector and each vector in `vectors`
        norm_vector = np.linalg.norm(vector)
        norm_vectors = np.linalg.norm(vectors, axis=1)

        # Handle zero vector case
        if norm_vector == 0:
            raise ValueError("Cosine similarity is not defined for a zero reference vector.")
        if np.any(norm_vectors == 0):
            raise ValueError("Cosine similarity is not defined for zero vectors in the input.")

        # Compute dot products
        dot_products = np.dot(vectors, vector)

        # Compute cosine similarities
        similarities = dot_products / (norm_vectors * norm_vector)

        return similarities

    def fit(self,curve):
        f = 1;
        if self.config["method"] == "leite":
            weights = curve.anchors;
            f = (weights**2 * curve.labels @ self.database[:len(curve.anchors),:]) / (weights**2 @ self.database[:len(curve.anchors),:]**2 )
        elif self.config["method"] == "rijn":
            weights = 2**curve.anchors;
            f = (weights * curve.labels @ self.database[:len(curve.anchors),:]) / (weights @ self.database[:len(curve.anchors),:]**2 )

        self.scaled = self.database * f

        dists = np.sum((self.scaled[:len(curve.anchors),:]
                    - curve.labels.reshape(-1,1))**2,0)

        dists = self.cossim(self.scaled[:len(curve.anchors),:].T,curve.labels)

        self.theta = np.argsort(dists)[::-1][:self.ntheta]
        self.curves = self.scaled[:,self.theta]
        self.status = "s"
        self._theta0 = self.theta



    def predict(self,xs):
        # Give the average of the k learning curves at xs
        return self.curves[xs.astype(int)-2,:].mean(1)
        # return self.curves[-1,:].mean()
        
        
