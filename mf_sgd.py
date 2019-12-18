import numpy as np
import time
from scipy.sparse import find

def loadData(filename):
    """
    params:
        filename: the path to the data file (ratings.dat)
    return:
        np array of the data that was being read
    """
    data = np.genfromtxt(filename, usecols=(0, 1, 2), delimiter='::', dtype='int')
    return(data)

def populateMatrix(data):
    '''
        params:
            data: the data of ratings
        returns:
            popularity matrix to be used in the training
    '''
    mat = np.zeros((6040, 3952), dtype=data.dtype)
    for index,element in enumerate(data):
        mat[element[0]-1, element[1]-1] = element[2]
    return(mat)

def matrixFactorization(data, nrFolds, k, seed, numIter, lr, penalty):
    '''
        params:
            data: the path to ratings data or variable
            nrFolds: the number of folds for cross-validation
            k: number of features
            seed: random seed values
            numIter: number of iterations for training
            lr: learning learning
            penalty: regulariozation parameter
        returns:
            RMSE and MAE values of each fold for both train and test sets
    '''

    np.random.seed(seed)

    seqs = [x % nrFolds for x in range(len(data))]
    np.random.shuffle(seqs)
    foldsRMSE_train = np.zeros(nrFolds)
    foldsMAE_train = np.zeros(nrFolds)
    foldsRMSE_test = np.zeros(nrFolds)
    foldsMAE_test = np.zeros(nrFolds)
    for fold in range(nrFolds):
        # Initialize train set
        trainSelect = np.array([x != fold for x in seqs])
        train = data[trainSelect]
        Xtrain = populateMatrix(train)
        nrUsers = Xtrain.shape[0]
        nrMovies = Xtrain.shape[1]
        # Random weights Initilization
        U = np.random.rand(nrUsers, k)
        M = np.random.rand(k, nrMovies)
        # Temp matrices Initialization
        Z = np.zeros((nrUsers, k))
        P = np.zeros((k, nrMovies))
        # Find Index of zero and non-zero elements
        (rows, cols, values) =  find(Xtrain)
        (nonRows, nonCols) = np.where(Xtrain==0)
        # Initialize arrays for error and X
        Xhat = np.zeros((nrUsers, nrMovies))
        errors = np.zeros((nrUsers, nrMovies))
        # Parameters and metrics

        RMSE_train = np.zeros((numIter))
        MAE_train = np.zeros((numIter))

        # train model for each fold
        beforeFold = time.time()
        for i in range(numIter):
            for index, element in enumerate(values):
                # X approximation
                Xhat[rows[index-1], cols[index]] = np.dot(U[rows[index-1],:], M[:,cols[index]])
                # Compute errors and gradients
                errors[rows[index-1], cols[index]] = element - Xhat[rows[index-1], cols[index]]
                gradientU = 2 * errors[rows[index-1], cols[index]] * M[:, cols[index]]
                gradientM = 2 * errors[rows[index-1], cols[index]] * U[rows[index-1], :]
                # Compute weights
                Z[rows[index-1],:] = U[rows[index-1],:] + lr * (gradientU  - penalty * U[rows[index-1] ,:] )
                P[:, cols[index]] = M[:, cols[index]] + lr * (gradientM  - penalty * M[:, cols[index]] )
            # Get predictions, truncate values, set non-existing indices to 0
            Xhat = np.dot(Z,P)
            Xhat[np.where(Xhat > 5)] = 5
            Xhat[np.where(Xhat < 1)] = 1
            Xhat[nonRows, nonCols] = 0
            U = Z
            M = P
            # Metrics evaluation
            RMSE_train[i] = np.sqrt(np.sum(errors**2)/len(values))
            MAE_train[i] = np.sum(np.abs(errors))/len(values)
            print('RMSE Train for iteration ' + str(i) + ' is ; ' + str(np.round(RMSE_train[i], 3)))
            print('MAE Train for iteration ' + str(i) + ' is ; ' + str(np.round(MAE_train[i], 3)) + '\n')
        foldsRMSE_train[fold] = RMSE_train[-1]
        foldsMAE_train[fold] = MAE_train[-1]
        afterFold = time.time()
        print('Time elapsed for fold ', fold, 'is ', round((afterFold - beforeFold)/60, 2), 'minutes \n')
        # Getting test set ready
        testSelect = np.array([x == fold for x in seqs])
        test = data[testSelect]
        Xtest = populateMatrix(test)
        (rows, cols, values) =  find(Xtest)
        (nonRows, nonCols) = np.where(Xtest==0)
        predictions = np.zeros((Xtest.shape[0], Xtest.shape[1]))
        # Get predictions on unseen data
        errorsTest = np.zeros((Xtest.shape[0], Xtest.shape[1]))
        for index, element in enumerate(values):
            predRat = np.dot(U[rows[index],:],M[:,cols[index]])
            predictions[rows[index],cols[index]] = predRat
            errorsTest[rows[index], cols[index]] = element - predRat

        predictions[np.where(predictions > 5)] = 5
        predictions[np.where(predictions < 1)] = 1
        predictions[nonRows, nonCols] = 0
        foldsRMSE_test[fold] = np.sqrt(np.sum(errorsTest**2)/len(values))
        foldsMAE_test[fold] = np.sum(np.abs(errorsTest))/len(values)
        print('RMSE test for fold ', fold, ' is ', round(foldsRMSE_test[fold], 3))
        print('MAE test for fold ', fold, ' is ', round(foldsMAE_test[fold], 3), '\n')
    print('Total RMSE for all the ', nrFolds, 'folds is ', round(np.mean(foldsRMSE_test), 3))
    print('Total MAE for all the ', nrFolds, 'folds is ', round(np.mean(foldsMAE_test), 3))
    return(foldsRMSE_test, foldsMAE_test, foldsRMSE_train, foldsMAE_train)


data = loadData("ratings.dat")
matrixFactorization(data, 5, 10, 1992, 75, 0.01, 0.05)
