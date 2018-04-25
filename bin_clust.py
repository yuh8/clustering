import numpy as np
import scipy as sp


class binclust(object):
    """Clustering binary data with two coding scheme"""

    def __init__(self, X, R=3):
        # X is a binary data frame
        self.X = X
        # Specifying number of clusters
        # Default is 3
        self.R = R

    def para_init(self):
        _num_col = self.X.shape[1]

    def E_step(self):
        pass
