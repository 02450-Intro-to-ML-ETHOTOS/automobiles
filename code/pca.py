from load_data import *

from scipy.linalg import svd
from matplotlib.pyplot import (figure, boxplot, plot, title, legend, xlabel, 
                               subplot, ylabel, show, xticks, ylim)

classLabels= raw_data.body_style.tolist()
dropped_data = raw_data.drop(['body_style', 'engine_type', 'num_of_cylinders', 'fuel_system',
                         'symboling', 'make',  'fuel_type', 'aspiration', 'num_of_doors',
                         'engine_location', 'drive_wheels','compression_ratio'], axis=1)
 
#Extract list with attributeNames
attributeNames = np.array(dropped_data.columns)

# Extract class names to python list,
# then encode with integers (dict)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(5)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Subtract mean value from data
X = np.array(dropped_data)
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

rho = (S*S) / (S*S).sum() 


