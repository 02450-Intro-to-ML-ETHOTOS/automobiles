import numpy as np
import pandas as pd

# n.b. paths are set as if executing from top repo dir, NOT the code dir
filename_attributes = "../data/imports-85.labels.txt"

#Specify path to file with attribute headers
filename_data = "../data/imports-85.data.txt"
names = np.loadtxt(filename_attributes, dtype=object)

#Read data into pandas data frame
raw_data = pd.read_csv(filename_data, names=names)

#Remove column "normalized-losses"
raw_data = raw_data.drop(['normalized_losses'], axis=1)

#Remove all rows with "?" as given value
raw_data = raw_data.replace('?', np.NaN)
raw_data = raw_data.dropna(axis=0)

# Cast numeric values to float
# see https://stackoverflow.com/a/16134561/13962373
raw_data = raw_data.apply(pd.to_numeric, errors='ignore')

