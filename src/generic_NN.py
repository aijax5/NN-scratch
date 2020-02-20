
import numpy as np
from src.NN import NN
import src.utils as utils
from datetime import datetime
# Settings
csv_filename = "data/tennis.csv"
 # seed for NN weight initialization

def getGenericNN(X,y,modelConfig):
    
    N, d = X.shape
    hidden_layers = modelConfig.hidden_layers # number of nodes in hidden layers i.e. [layer1, layer2, ...]
    eta = modelConfig.eta # learning rate
    n_epochs = modelConfig.n_epochs# number of training epochs
    n_folds = modelConfig.n_folds# number of folds for cross-validation
    seed_crossval =modelConfig.seed  # seed for cross-validation
    n_classes = modelConfig.n_classes

    print(" -> X.shape = {}, y.shape = {}, n_classes = {}\n".format(X.shape, y.shape, n_classes))
    print("Neural network model:")
    print(" input_dim = {}".format(d))
    print(" hidden_layers = {}".format(hidden_layers))
    print(" output_dim = {}".format(n_classes))
    print(" eta = {}".format(eta))
    print(" n_epochs = {}".format(n_epochs))
    print(" n_folds = {}".format(n_folds))
    print(" seed_crossval = {}".format(seed_crossval))

    # Create cross-validation folds
    idx_all = np.arange(0, N)
    idx_folds = utils.crossval_folds(N, n_folds, seed=datetime.now()) # list of list of fold indices

    # Train/evaluate the model on each fold
    acc_train, acc_valid = list(), list()
    print("Cross-validating with {} folds...".format(len(idx_folds)))
    for i, idx_valid in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_valid)
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        # Build neural network classifier model and train
        model = NN(input_dim=d, output_dim=n_classes,
                hidden_layers=hidden_layers)
        model.train(X_train, y_train, eta=eta, n_epochs=n_epochs)

        # Make predictions for training and test data
        ypred_train = model.predict(X_train)
        ypred_valid = model.predict(X_valid)

        # Compute training/test accuracy score from predicted values
        acc_train.append(100*np.sum(y_train==ypred_train)/len(y_train))
        acc_valid.append(100*np.sum(y_valid==ypred_valid)/len(y_valid))

        # Print cross-validation result
        print(" Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_valid = {})".format(
            i+1, n_folds, acc_train[-1], acc_valid[-1], len(X_train), len(X_valid)))

    # Print results
    print("""*****************************FINAL GENERIC MODEL RESULTS ********************\n  
    -> acc_train_avg = {:.2f}%, acc_valid_avg = {:.2f}%""".format(sum(acc_train)/float(len(acc_train)), sum(acc_valid)/float(len(acc_valid))))
    
    return model