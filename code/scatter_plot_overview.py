from load_data import *

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import os

# make output dir
os.makedirs("../out/plots", exist_ok=True)

class_labels = raw_data["body_style"].values # 193 labels

#then encode with integers (dict)
class_names = sorted(set(class_labels))
class2idx = dict(zip(class_names, range(len(class_names))))
idx2label = dict(zip(range(len(class_names)), [c.capitalize() for c in class_names]))

# Extract vector y, convert to NumPy array
y = np.asarray([class2idx[value] for value in class_labels])


# subset dataframe to numerical attributes
X = raw_data[numerical_attrs].values
C = len(class_names)
M = X.shape[1] # num attributes
N = X.shape[0] # num obs

# Plot scatter plots

## Next we plot a number of atttributes
Attributes = [i for i in range(len(numerical_attrs))]
NumAtr = len(Attributes)

fig = plt.figure(figsize=(14,14))
# nested for-loop to generate attribute vs attribute
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        plt.subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        
        # iterate over classes and plot each individually
        for c in range(C):
            class_mask = (y==c)
            plt.scatter(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], s=2.0, marker='.', label=idx2label[c])
            
        # handle x-axis labels
        if m1 == 0: # add label to top row plots
            plt.xlabel(numerical_attrs[Attributes[m2]], rotation=45)
            fig.axes[m2].xaxis.set_label_position("top")
            plt.xticks([]) # no ticks
        elif m1 == (NumAtr - 1): # add ticks to bottom row plots
            plt.xlabel("")
            plt.xticks(rotation=45)
        else:
            plt.xticks([]) # no ticks
        
        # handle y-axis labels
        if m2 == 0: # add labels and ticks to leftmost column
            plt.ylabel("")
        elif m2 == (NumAtr - 1):
            plt.ylabel(numerical_attrs[Attributes[m1]], rotation=45, labelpad=33)
            ax_i = (m1*NumAtr+NumAtr-1) # calculate index of fig ax
            fig.axes[ax_i].yaxis.set_label_position("right")
            plt.yticks([]) # no ticks
        else:
            plt.yticks([]) # no ticks
        #ylim(0,X.max()*1.1)
        #xlim(0,X.max()*1.1)


handles, labels = fig.axes[0].get_legend_handles_labels() # get handles and labels for very first subplot
fig.legend(handles, labels, title='Body Style', loc="upper right") # use handles and labels from first subplot to make legend

plt.suptitle("Automobiles - Combinations of Numerical Attributes", va="bottom", fontsize="xx-large")

# show or save plot
# plt.show()
plt.savefig("../out/plots/scatterplot_overview.png", dpi=200)
