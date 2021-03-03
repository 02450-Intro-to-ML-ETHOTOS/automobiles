from pca import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

i = 5
j = 13

# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)

plot(X[:, i], X[:, j], 'o')

# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
title('Body Style Data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Ouput result to screen
show()
