# goal: estimate the generalization error for different values of h
# use 2-level (K1=K2=10) cross-validation for optimal value of the number 
#of hidden units

#normalized and one-hot encoded data imported from regression_transform_data
from regression_transform_data import *

import matplotlib.pyplot as plt
#from scipy.io import loadmat
#from scipy import stats
import torch
from toolbox_02450 import train_neural_net, draw_neural_net

attributeNames= names.tolist()      #cast to numpy list
C = 2                               #Number of classes.
n_hidden_units = 5                  # number of hidden units
n_replicates = 1                    # number of networks trained in each k-fold
max_iter = 10000

K = 10    # Number of folds
CV = model_selection.KFold(K, shuffle=True)
n_out = 1

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model
# With regression, we do not apply a transfer-function to the final layer
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, n_out), # n_hidden_units to output neurons
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn = torch.nn.MSELoss() #Network not trained based on a cross entropy loss, 
#but on a mean-square-error-loss (MSE loss).

print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    