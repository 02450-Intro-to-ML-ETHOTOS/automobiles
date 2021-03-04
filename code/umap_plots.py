from load_data import *

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

sumstats_numerical_attrs = raw_data.describe()
numerical_attrs = sumstats_numerical_attrs.columns.values

class_labels = raw_data["body_style"].values # 193 labels

#then encode with integers (dict)
class_names = sorted(set(class_labels))
class_dict = dict(zip(class_names, range(len(class_names))))

# Extract vector y, convert to NumPy array
y = np.asarray([class_dict[value] for value in class_labels])

# subset dataframe to numerical attributes
X = raw_data[numerical_attrs].values


reducer = umap.UMAP()
scaled_data = StandardScaler().fit_transform(X)
embedding = reducer.fit_transform(scaled_data)
print(embedding.shape)

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    # c=[sns.color_palette()[x] for x in class_dict])
    c=[sns.color_palette()[x] for x in y])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Automobiles Numerical Attributes', fontsize=24)
plt.show()
