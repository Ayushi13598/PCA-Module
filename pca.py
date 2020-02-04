import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as prca

class PCA:
    def __init__(self,n_component=2):
        self.n_component = n_component
        self.variance_ratio = None
        self.modl = prca(n_component)
    def fit(self,x):
        self.modl.fit(x)
        self.variance_ratio = self.modl.explained_variance_ratio_
    def transform(self,x):
        return self.modl.fit_transform(x)
