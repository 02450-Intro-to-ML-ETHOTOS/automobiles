
from classification_transform_data import *

from sklearn import tree

# Fit regression tree classifier, Gini split criterion, no pruning
criterion='gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=5)
dtc = dtc.fit(X,y)

fname='tree_' + criterion + '_car_data'
# Export tree graph .gvz file to parse to graphviz
out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)

# Extracting first row representing type "convertible"
x = (X[0,:]).reshape(1,-1)	

#
# Evaluate the classification tree for the new data object
x_class = dtc.predict(x)[0].reshape(-1,1)	
mask = [a for a in x_class[:,0] == 1]

# Print and verify results
print('\nNew object attributes:')
print(dict(zip(attributeNames,x[0])))
print('\nClassification result:')
print(class_names[int(mask == 'true')-1])