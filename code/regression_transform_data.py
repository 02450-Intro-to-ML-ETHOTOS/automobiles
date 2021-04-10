from load_data import *

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# extract y and convert to np array
y = raw_data["price"].to_numpy()
N = len(y)

# transform numerical data to get mean 0 and unit variance
numerical_attrs = [a for a in numerical_attrs if a !=
                   "price"]  # 13 attributes (14 incl. price)
X_num = raw_data[numerical_attrs].values
X_num = X_num - (np.ones((N, 1)) * X_num.mean(axis=0))
X_num = X_num * (1/np.std(X_num, 0))
# print(X_num[:5, :])

# one-hot encode categorical data
X_cat = raw_data[categorical_attrs]  # 11 attributes
# infer categories from values by casting to category
X_cat = X_cat.astype('category')
# print(X_cat["symboling"])

# commnet/uncomment to control which variables are included
categorical_attrs = ["aspiration", 
                    "body_style", 
                    "drive_wheels", 
                    "engine_location", 
                    "engine_type", 
                    "fuel_system", 
                    "fuel_type", 
                    "make", 
                    "num_of_cylinders", 
                    "num_of_doors", 
                    "symboling"]

# 50 attributes, when using one-hot encoding
# N.B. if we don't drop first, the coefficients may explode. See "Dummy variable trap"
X_cat = pd.get_dummies(X_cat[categorical_attrs], columns=categorical_attrs, drop_first=True)
# print(X_cat.head()) # verify one-hot encoding
# print(X_cat.shape)

# combine numerical and categorical data
X = np.hstack((X_num, X_cat.values))  # 63 attributes in total

# print(X)
# print(X.shape)

# set up useful variables
N, M = X.shape

#List of attribute names
attributeNames = numerical_attrs + (X_cat.columns.values).tolist()