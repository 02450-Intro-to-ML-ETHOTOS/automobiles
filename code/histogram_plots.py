from load_data import *

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as sps


sumstats_numerical_attrs = raw_data.describe()
numerical_attrs = sumstats_numerical_attrs.columns.values

# subset dataframe to X
X = raw_data[numerical_attrs].values
C = len(np.unique(raw_data["body_style"]))
M = X.shape[1] # num attributes
N = X.shape[0] # num obs


# Plot histograms
plt.figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    plt.subplot(u,v,i+1)
    plt.hist(X[:,i])
    # plt.xlabel(numerical_attrs[i])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    plt.title(numerical_attrs[i])

plt.suptitle("Automobiles, numerical attributes. No normalization, no outlier removal")  # 
plt.show()

