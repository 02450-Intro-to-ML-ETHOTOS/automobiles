import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# n.b. paths are set as if executing from the code dir
# Specify path to file with attribute headers
filename_attributes = "../data/imports-85.labels.txt"
names = np.loadtxt(filename_attributes, dtype=object)

# Read data into pandas data frame
filename_data = "../data/imports-85.data.txt"
raw_data = pd.read_csv(filename_data, names=names)

# Clean data
# Remove column "normalized-losses"
raw_data = raw_data.drop(['normalized_losses'], axis=1)

# Remove missing values
# Remove all rows with "?" as given value
raw_data = raw_data.replace('?', np.NaN)
raw_data = raw_data.dropna(axis=0)

# Cast numeric values to float to avoid type problems
# see https://stackoverflow.com/a/16134561/13962373
raw_data = raw_data.apply(pd.to_numeric, errors='ignore')

# set up attributes as global variables
numerical_attrs = ["wheel_base", 
                    "length", 
                    "width", 
                    "height", 
                    "curb_weight", 
                    "engine_size", 
                    "bore", 
                    "stroke", 
                    "compression_ratio", 
                    "horsepower", 
                    "peak_rpm", 
                    "city_mpg", 
                    "highway_mpg", 
                    "price"]

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

# setup dicts for easy lookup
class_names = sorted(set(raw_data["body_style"].values)) # Convertible, Hardtop, Hatchback, Sedan, Wagon
class2idx = dict(zip(class_names, range(len(class_names)))) # e.g. class2idx["convertible"] = 0
idx2Class = dict(zip(range(len(class_names)), [c.capitalize() for c in class_names])) # n.b. capitalized!

num_attr2idx = dict(zip(numerical_attrs, range(len(numerical_attrs)))) # e.g. num_attr2idx["length"] = 1
