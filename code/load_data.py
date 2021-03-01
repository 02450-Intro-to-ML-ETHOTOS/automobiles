import numpy as np
import pandas as pd

# n.b. paths are set as if executing from top repo dir, NOT the code dir
filename_attributes = "../data/imports-85.labels.txt"
filename_data = "../data/imports-85.data.txt"

names = np.loadtxt(filename_attributes, dtype=str)

df = pd.read_csv(filename_data, names=names)
print(df.head())
