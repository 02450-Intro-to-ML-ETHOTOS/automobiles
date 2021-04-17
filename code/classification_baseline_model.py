from classification_transform_data import *
# imports: numpy as np, pandas as pd
# import numpy as np


class ClassificationBaselineModel(object):
    """A simple baseline model which simply predicts the class with highest frequency"""
    def __init__(self):
        self.y_pred = None

    def fit(self, y):
        # TODO: use K-fold CV?
        # count occourences of each class
        counts = y.sum(axis=0)

        # the most common class is the one with largest count
        class_common = np.argmax(counts)
        
        self.y_pred = class_common

    def predict(self, X):
        assert(self.y_pred is not None), "Model not trained yet!"
        
        y_pred = np.array([self.y_pred] * X.shape[0])
        return y_pred.squeeze()


# test      
# bm = ClassificationBaselineModel()
# bm.fit(y)
# print(bm.predict(X))
