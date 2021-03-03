from pca import *

#Extract Attributes wheel-base, length, width, height
filtered_data = dropped_data.iloc [:, [0,1,2,3]]
attributeNames = np.array(filtered_data.columns)
Xplot = np.array(filtered_data)

#------------------------------------------------------------------------------
### Boxplot of selected colums
boxplot(Xplot)
r = np.arange(1,Xplot.shape[1]+1)
xticks(r,attributeNames)
xticks(rotation = 45)
ylabel('m')
title('Automobiles - boxplot')
show()
#------------------------------------------------------------------------------

figure(figsize=(14,7))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c
    
    boxplot(Xplot[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    title('Class: '+classNames[c])
    xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
    y_up = Xplot.max()+(Xplot.max()-Xplot.min())*0.1; y_down = Xplot.min()-(Xplot.max()-Xplot.min())*0.1
    ylim(y_down, y_up)
show()