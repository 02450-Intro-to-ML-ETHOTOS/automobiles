from classification_transform_data import *

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from scipy.io import loadmat
from sklearn import model_selection, tree


classes = [np.argmax(enc) for enc in y]
y = np.array(classes)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2,22,1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train= X[train_index,:], y[train_index]
    X_test, y_test= X[test_index,:], y[test_index]
   
    # swap row - column
    #y_test = np.concatenate((y_test[:,0].reshape((-1,1)).T,y_test[:,1].reshape((-1,1)).T,
    #                         y_test[:,2].reshape((-1,1)).T, y_test[:,3].reshape((-1,1)).T,
    #                         y_test[:,4].reshape((-1,1)).T), axis = 0)
    
    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini/Entropy split criterion, different pruning level
        # can also use "min_samples_split" and "min_samples_leaf" as stop criteria
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t, class_weight = 'balanced')
        #dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=t)
        #dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=t) 
        dtc = dtc.fit(X_train,y_train)
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1

# calculate mean error over k folds for each split/optimization criterion
error_train_mean = np.mean(Error_train, axis=1)
error_test_mean = np.mean(Error_test, axis=1)

# finally, choose max_depth
# min of mean validation error over K folds is optimal
opt_val_err = np.min(error_test_mean)
# the lambda that gives opt_val_err
opt_criteria_idx = np.argmin(error_test_mean)
opt_criteria = tc[opt_criteria_idx]

dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_criteria, class_weight = 'balanced')
dtc.fit(X, y) # N.B. we train with ALL data

f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show()
