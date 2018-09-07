# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:04:05 2018

@author: Ryan Farr (far231)
"""

import pandas as pd
from sklearn.decomposition import PCA

def pca2comp(data, features):
    """
    data = pd dataframe that only contains the features to be reduced (ie no labels or extraneous data)
    
    features = labels that will be used to differentiate samples. Can come from the same pd df but must be sliced.
    """

    pca = PCA(n_components=2, svd_solver='full')
    pca_info = pca.fit(data)
    T = pca.transform(data)
    full_pca = pd.DataFrame(data=T, columns=['component1', 'component2'], index = data.index)
    full_pca['features'] = features
    return print(pca_info, "PCA analysis is called full_pca")



