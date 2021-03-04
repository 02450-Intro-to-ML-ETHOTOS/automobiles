from pca import *
from scipy.stats import zscore

#Extract Attributes wheel-base, length, width, height
#dropped_data = dropped_data.iloc [:, [0,1,2,3]]
#attributeNames = np.array(dropped_data.columns)
X = np.array(dropped_data)

#The data matrix should be standardized to have zero mean and 
#unit standard deviation
X  = zscore(X, ddof=1)

#------------------------------------------------------------------------------
### BoX of selected colums
figure(figsize=(10,7))
boxplot(X)
r = np.arange(1,X.shape[1]+1)
xticks(r,attributeNames)
xticks(rotation = 45)
ylabel('m')
title('Automobiles - boxplot')
show()
#------------------------------------------------------------------------------

price_plot = dropped_data.iloc[:, [12]]
attributeName = np.array(price_plot.columns)
price_plot = np.array(price_plot)
### Boxplot of prices
boxplot(price_plot)
r = np.arange(1,price_plot.shape[1]+1)
xticks(r,attributeName)
xticks(rotation = 45)
ylabel('dollar')
title('Automobiles - boxplot')
show()
#------------------------------------------------------------------------------

figure(figsize=(25,7))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c
    
    boxplot(X[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    title('Class: '+classNames[c])
    xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
    y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
    ylim(y_down, y_up)
show()