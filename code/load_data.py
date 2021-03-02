import numpy as np
import pandas as pd

# n.b. paths are set as if executing from top repo dir, NOT the code dir
filename_attributes = "../data/imports-85.labels.txt"

#Specify path to file with attribute headers
filename_data = "../data/imports-85.data.txt"
names = np.loadtxt(filename_attributes, dtype=str)

#Read data into pandas data frame
raw_data = pd.read_csv(filename_data, names=names)

#Remove column "normalized-losses"
data = raw_data.drop(['normalized-losses'], axis=1)

#Remove all rows with "?" as given value
data = raw_data.replace('?', np.NaN)
data = raw_data.dropna(axis=0)

