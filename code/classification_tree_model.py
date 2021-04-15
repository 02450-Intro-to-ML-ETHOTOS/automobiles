from classification_transform_data import *

from sklearn import tree

# Fit regression tree classifier, Gini split criterion, no pruning
criterion='gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=100)
dtc = dtc.fit(X,y)

fname='tree_' + criterion + '_car_data'
# Export tree graph .gvz file to parse to graphviz
out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)

