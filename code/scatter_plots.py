from load_data import *

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as sps


sumstats_numerical_attrs = raw_data.describe()
numerical_attrs = sumstats_numerical_attrs.columns.values

class_labels = raw_data["body_style"].values # 193 labels
# TODO: encode as y

#then encode with integers (dict)
class_names = sorted(set(class_labels))
class_dict = dict(zip(class_names, range(len(class_names))))

# Extract vector y, convert to NumPy array
y = np.asarray([class_dict[value] for value in class_labels])


# subset dataframe to numerical attributes
X = raw_data[numerical_attrs].values
C = len(class_names)
M = X.shape[1] # num attributes
N = X.shape[0] # num obs

# Plot scatter plots

## Next we plot a number of atttributes
Attributes = [i for i in range(len(numerical_attrs))]
# Attributes = [0, 1, 2, 3, 4]
NumAtr = len(Attributes)

plt.figure(figsize=(12,12))
# nested for-loop to generate attribute vs attribute
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        plt.subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        
        # iterate over classes and plot each individually
        for c in range(C):
            class_mask = (y==c)
            plt.plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            
            if m1==NumAtr-1: # add label to bottom row plots
                plt.xlabel(numerical_attrs[Attributes[m2]], rotation=45)
            else:
                plt.xticks([])
            if m2==0: # add label to leftmost column plots
                plt.ylabel(numerical_attrs[Attributes[m1]])
            else:
                plt.yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
# plt.legend(class_names)
plt.show()
