import numpy as np
from src.NN import NN
import src.utils as utils
import copy
import random
from box import Box 
from datetime import datetime
# if(__name__ == 'main')
# Settings

def split_learn(X):
    datasets,features=X.shape
    
    splitWeights=list()
    splits = np.hsplit(X,features)
    # return splits
    print("splits made = ",len(splits))
    for o,oX in enumerate(splits):
        N, d = oX.shape

        print("Neural network model:")
        print(" input_dim = {}".format(d))
        print(" hidden_layers = {}".format(hidden_layers))
        print(" output_dim = {}".format(n_classes))
        print(" eta = {}".format(eta))
        print(" n_epochs = {}".format(n_epochs))
        print(" n_folds = {}".format(n_folds))
        print(" seed_crossval = {}".format(seed_crossval))
        # print(" seed_weights = {}\n".format)

        # Create cross-validation folds
        idx_all = np.arange(0, N)
        idx_folds = utils.crossval_folds(N, n_folds, seed=seed_crossval) # list of list of fold indices
        
        # Train/evaluate the model on each fold
        acc_train, acc_valid = list(), list()
        print("Cross-validating with {} folds...".format(len(idx_folds)))
        for i, idx_valid in enumerate(idx_folds):
        #  seed=seed_weights
            # Collect training and test data from folds
            idx_train = np.delete(idx_all, idx_valid)
            X_train, y_train = oX[idx_train], y[idx_train]
            X_valid, y_valid = oX[idx_valid], y[idx_valid]

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
            print(" Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_valid {})".format(i+1,n_folds, acc_train[-1], acc_valid[-1], len(X_train), len(X_valid)))

        # Print results
        print(o," : ")
        print("  -> acc_train_avg = {:.2f}%, acc_valid_avg = {:.2f}%".format(sum(acc_train)/float(len(acc_train)), sum(acc_valid)/float(len(acc_valid))))

        splitWeights.append(model.get_weights())
        
    
    return splitWeights

def get_aggregate_weights(wt):
    new=[]
    
    print("wth",new)
    rep = copy.deepcopy(wt)
    for i,mx in enumerate(rep):
        # m2 = wt[i]
        if i==0:
            print("hitting new")
            new = mx
            print(new[1])
        else:
            for j,lx in enumerate(mx):
                if j==0 and i!=0:
                    # print(new[j],"88",lx)
                    
                    new[j] = [np.append(new[j],lx)]
                    print(len(new[j]),"*******************")
                else:
                    new[j] = np.add(new[j],lx)
                    print(len(new[j]))

    for k in range(len(new)):
        if not k == 0:
            print(k)
            new[k] = np.divide(new[k],len(wt))
            # print(type(l1))
    print( new )
    return new


def get_aggregate_model(X,newWeights):
    # for oX in enumerate(splits):
    # print("Reading '{}'...".format(csv_filename))
    # X, y, n_classes = utils.read_csv(csv_filename, target_name="y", normalize=True)
    N, d = X.shape

    print("Neural network model:")
    print(" input_dim = {}".format(d))
    print(" hidden_layers = {}".format(hidden_layers))
    print(" output_dim = {}".format(n_classes))
    print(" eta = {}".format(eta))
    print(" n_epochs = {}".format(n_epochs))
    print(" n_folds = {}".format(n_folds))
    print(" seed_crossval = {}".format(seed_crossval))

    # list of list of fold indices
    model = NN(input_dim=d, output_dim=n_classes,
                        hidden_layers=hidden_layers,AvgWeights=newWeights)
    # Train/evaluate the model on each fold
    return model


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
    idx_folds = utils.crossval_folds(N, n_folds, seed=seed_crossval) # list of list of fold indices

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


if __name__ == "__main__" :
    # seed for cross-validation
    csv_filename = "data/tennis.csv"
    print("Reading '{}'...".format(csv_filename))
    X, y, n_c = utils.read_csv(csv_filename, target_name="y", normalize=True)

    config = {
        "n_classes" : n_c,
        "hidden_layers" : [1,2], # number of nodes in hidden layers i.e. [layer1, layer2, ...],
        "eta" : 0.1, # learning rate,
        "n_epochs" : 50, # number of training epochs,
        "n_folds" : 4 ,# number of folds for cross-validation,
        "seed_crossval" : 1 
    }
    config = Box(config)
    N,d = X.shape
    fold = utils.crossval_folds(N, 5, seed=datetime.now()) # list of list of fold indices
    valid = random.randrange(0,len(fold))
    validSets = fold[valid]
    allSets = np.arange(0, N)
    trainSets = np.delete(allSets, validSets)

    nX, nY = X[trainSets], y[trainSets]
    vX, vY = X[validSets], y[validSets]
    
    genericModel = genericModel(nX, nY, config)
    newModelWeights = split_learn(nX)
    aggregateWeights = get_aggregate_weights(newModelWeights)
    newModel = get_aggregate_model(nX,aggregateWeights)
    print("new weights of the aggreagated model are:", newModel.get_weights())