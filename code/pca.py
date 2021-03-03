from load_data import *

from scipy.linalg import svd

classLabels= raw_data.body_style.tolist()
X = raw_data.drop(['body_style', 'engine_type', 'num_of_cylinders', 'fuel_system',
                         'symboling', 'make',  'fuel_type', 'aspiration', 'num_of_doors',
                         'engine_location', 'drive_wheels'], axis=1)
 
#Extract list with attributeNames
attributeNames = np.array(X.columns)

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
X = np.array(X)
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

