# -*- coding: utf-8 -*-
from regression_transform_data import *

# extract body style columns and then remove them from matrix of 
# categorical values
body_style = X_cat[["body_style_hardtop", "body_style_hatchback", "body_style_sedan",
                    "body_style_wagon"]]
X_cat = X_cat.drop(body_style, axis=1)

# convert body_style dataframe to numpy array
y = body_style.to_numpy()

# add body_style covertible to class matrix
convertible = (1*np.all(y == 0, axis=1)).reshape((-1,1))
y = np.append(y, convertible, axis=1)

# append numerical attributes header list with price, (now 14 in total)
numerical_attrs.append("price")

#remove categorical attributes header list with "body_style" (now 10 in total)
categorical_attrs.remove("body_style")

# transform numerical data of price to get mean 0 and unit variance
X_num = raw_data[numerical_attrs].values
X_num = X_num - (np.ones((N, 1)) * X_num.mean(axis=0))
X_num = X_num * (1/np.std(X_num, 0))

# combine numerical and categorical data
X = np.hstack((X_num, X_cat.values))  # 60 attributes in total

# set up useful variables
N, M = X.shape

# list of attribute names
attributeNames = numerical_attrs + (X_cat.columns.values).tolist()


