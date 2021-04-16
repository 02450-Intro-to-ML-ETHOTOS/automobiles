from classification_transform_data import *

from sklearn import model_selection, tree

def onehot2classidx(y):
    classes = [np.argmax(enc) for enc in y]
    return np.array(classes)

class ClassificationTreeModel(object):
    """A multi-class classification tree model"""
    def __init__(self):
        self.dtc = None
        self.criteria_opt = None
        
    def fit(self, X, y, criteria, K):
        if len(y.shape) != 1:
            # sklearn linear_model expects no one-hot encoding
            y = onehot2classidx(y)
        
        # Tree complexity parameter - constraint on maximum depth
        tc = criteria

        CV = model_selection.KFold(n_splits=K,shuffle=True)

        # Initialize variable
        Error_train = np.empty((len(tc),K))
        Error_test = np.empty((len(tc),K))

        k=0
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train, y_train= X[train_index,:], y[train_index]
            X_test, y_test= X[test_index,:], y[test_index]

            for i, t in enumerate(tc):
                # Fit decision tree classifier, Gini/Entropy split criterion, different pruning level
                # can also use "min_samples_split"/"min_samples_leaf" as stop criteria
                dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t, class_weight = 'balanced')
                dtc = dtc.fit(X_train,y_train)
                y_est_test = dtc.predict(X_test)
                y_est_train = dtc.predict(X_train)
                # Evaluate misclassification rate over train/test data (in this CV fold)
                misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
                misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
                Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
            k+=1

        # calculate mean error over k folds for each split/optimization criterion
        error_train_mean = np.mean(Error_train, axis=0)
        error_test_mean = np.mean(Error_test, axis=0)

        # finally, choose best criteria based on min of mean validation error over K folds
        opt_val_err = np.min(error_test_mean)
        # the criteria that gives opt_val_err
        opt_criteria_idx = np.argmin(error_test_mean)
        opt_criteria = tc[opt_criteria_idx]

        # fit model again, using optimal criteria
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_criteria, 
                                          class_weight = 'balanced')
        dtc.fit(X, y) # N.B. we train with ALL data

        self.dtc = dtc
        self.criteria_opt = opt_criteria

        return opt_criteria, opt_criteria_idx, error_train_mean, error_test_mean

    def predict(self, X):
        assert(self.dtc is not None), "Model not trained yet!"
        
        return self.dtc.predict(X)

#criteria = np.arange(2,20,1)   
#tree_model = ClassificationTreeModel()    
#fit = tree_model.fit(X, y, criteria, 10)    

