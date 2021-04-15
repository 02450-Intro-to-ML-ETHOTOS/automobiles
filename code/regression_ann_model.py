from regression_transform_data import *
# imports: numpy as np, pandas as pd
import torch


def train_ann(model, loss_func, X, y, max_iter):
    # Make a new net (calling model() makes a new initialization of weights)
    net = model()

    # initialize weights based on limits that scale with number of in- and
    # outputs to the layer, increasing the chance that we converge to
    # a good solution
    torch.nn.init.xavier_uniform_(net[0].weight)
    torch.nn.init.xavier_uniform_(net[2].weight)

    optimizer = torch.optim.Adam(net.parameters())

    learning_curve = []  # setup storage for loss at each step
    loss_final = 1e6
    logging_frequency = 1000  # display the loss every 1000th iteration

    print("Training\n\tIter.\tLoss")
    for i in range(max_iter):
        y_est = net(X)  # forward pass, predict labels on training set
        loss = loss_func(y_est, y)  # determine loss
        # do backpropagation of loss and optimize weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save values and print for log
        loss_value = loss.data.numpy()  # get numpy array instead of tensor
        learning_curve.append(loss_value)  # record loss for later display
        loss_final = loss_value  # update final loss

        # display loss with some frequency:
        if (i != 0) & ((i+1) % logging_frequency == 0):
            msg = f"\t{str(i+1)}/{max_iter}\t{loss_value}"
            print(msg)

    print("\tFinal loss:")
    msg = f"\t{str(i+1)}\t{loss_value}"
    print(msg)

    return net, loss_final, learning_curve


class RegressionANNModel(object):
    """An Artificial Neural Network model for regression"""

    def __init__(self):
        self.net = None
        self.n_hidden = None

    def fit(self, X, y, hidden_list, K, max_iter=10000):
        N, M = X.shape

        # the number of models
        S = len(hidden_list)

        # matrices for storing coefficients and errors
        ws = np.empty((K, S, M))
        train_error = np.empty((K, S))
        test_error = np.empty((K, S))

        # for each value [of lambda] use K = 10 fold cross-validation to estimate the
        # generalization error

        # set up cross validation
        CV = model_selection.KFold(K, shuffle=True, random_state=42)

        # define loss function once
        loss_fn = torch.nn.MSELoss()  # we train with mean-square-error-loss (MSE loss)

        nets = []  # list to hold the models for later selection
        errors = []  # make a list for storing generalizaition error in each loop

        # do cross-validation steps
        # Iterate over k=1,...,K splits
        for k, (train_index, test_index) in enumerate(CV.split(X, y)):
            # print(f"CV Fold: {k+1}/{K}")

            # Let Dk^train, Dk^test be the k'th split of D
            # extract training and test set for current CV fold
            # also convert to tensors
            X_train = X[train_index, :]
            y_train = y[train_index].reshape((-1, 1))
            X_test = X[test_index, :]
            y_test = y[test_index].reshape((-1, 1))

            for s, n_hidden in enumerate(hidden_list):
                # Train model Ms on the data Dk^train

                # Define the model
                # With regression, we do not apply a transfer-function to the final layer
                def model(): return torch.nn.Sequential(
                    # M features to n_hidden_units
                    torch.nn.Linear(M, n_hidden),
                    # torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.ReLU(),
                    # n_hidden_units to output neurons
                    torch.nn.Linear(n_hidden, 1),
                    # no final tranfer function, i.e. "linear output"
                )

                net, final_loss, learning_curve = train_ann(
                    model, loss_fn, X_train, y_train, max_iter)

                # save net
                nets.append(net)

                # evaluate model
                # y_pred_train = net(X_train) # TODO: not needed?
                y_test_pred = net(X_test).detach().numpy()

                y_delta_test = y_test.numpy() - y_test_pred
                error_test = (y_delta_test.T @ y_delta_test) / \
                    N  # TODO: alternative?
                # error_test = y_delta_test**2

                # store results
                # recall: k=fold, s=model
                # train_error[k, s] = error_train
                test_error[k, s] = error_test

        # calculate mean error over k folds for each lambda
        error_train_mean_per_lambda = np.mean(train_error, axis=0)
        error_test_mean_per_lambda = np.mean(test_error, axis=0)

        # finally, choose optimal parameter
        opt_val_err = np.min(error_test_mean_per_lambda)
        # the parameter that gives opt_val_err
        opt_param_idx = np.argmin(error_test_mean_per_lambda)
        opt_param = hidden_list[opt_param_idx]

        # fit model again, using optimal parameter and all training data
        def model(): return torch.nn.Sequential(
            torch.nn.Linear(M, opt_param),  # M features to n_hidden_units
            # torch.nn.Tanh(),   # 1st transfer function,
            torch.nn.ReLU(),
            torch.nn.Linear(opt_param, 1),  # n_hidden_units to output neurons
            # no final tranfer function, i.e. "linear output"
        )

        print(
            f"Opt param {opt_param} - training model with optimal parameter on all data")

        # N.B. using full dataset
        net, final_loss, learning_curve = train_ann(
            model, loss_fn, X, y.reshape((-1, 1)), max_iter)

        # save best model
        self.net = net
        self.n_hidden = opt_param

        return opt_param, opt_param_idx, learning_curve, error_train_mean_per_lambda, error_test_mean_per_lambda

    def predict(self, X):
        assert(self.net is not None), "Model not trained yet!"

        y_pred = self.net(X)
        return y_pred.detach().numpy().squeeze()


# test
# X = torch.Tensor(X)
# y = torch.Tensor(y)
# ann_model = RegressionANNModel()
# ann_model.fit(X, y, hidden_list=[1, 2, 16], K=2, max_iter=1000)
# print(ann_model.predict(X[:3,:]))
