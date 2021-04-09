from regression_transform_data import *
# imports: numpy as np, pandas as pd


class RegressionBaselineModel(object):
    """A simple baseline model which simply predicts the mean of y"""
    def __init__(self):
        self.y_pred = None

    def fit(self, y):
        # TODO: use K-fold CV
        self.y_pred = y.mean()

    def predict(self, X):
        assert(self.y_pred is not None), "Model not trained yet!"
        
        y_pred = np.ones(X.shape[0]) * self.y_pred
        return y_pred

    
# test      
# rbm = RegressionBaselineModel()
# rbm.fit(y)
# print(rbm.predict(X))
