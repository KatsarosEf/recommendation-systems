import numpy as np
from scipy.sparse import find
import time

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


def matrixFactorizationALS(data, nrFolds, k, seed, numIter, penalty):
    '''
        params:
            data: the path to ratings data or variable
            nrFolds: the number of folds for cross-validation
            k: number of features
            seed: random seed values
            numIter: number of iterations for training
            penalty: regulariozation parameter
        returns:
            RMSE and MAE values of each fold for both train and test sets
    '''
    np.random.seed(seed)
    # Find movies that have not been rated by anyone
    zeroCols =  np.where(~populateMatrix(data).any(axis=0))
    seqs = [x % nrFolds for x in range(len(data))]
    np.random.shuffle(seqs)
    RMSEtest = np.zeros((nrFolds))
    MAEtest = np.zeros((nrFolds))
    RMSEtrain = np.zeros((nrFolds))
    MAEtrain = np.zeros((nrFolds))

    RMSE = np.zeros((nrFolds, numIter))
    MAE = np.zeros((nrFolds, numIter))
    timeFold = np.zeros(nrFolds)
    for fold in range(nrFolds):
        beginTime = time.time()
        # Initialize train set
        trainSelect = np.array([x != fold for x in seqs])
        train = data[trainSelect]
        Xtrain = populateMatrix(train)
        Xtrain = np.delete(Xtrain, zeroCols, 1)
        # Initialize test set
        testSelect = np.array([x == fold for x in seqs])
        test = data[testSelect]
        Xtest = populateMatrix(test)
        Xtest = np.delete(Xtest, zeroCols, 1)

        nrUsers = Xtrain.shape[0]
        nrMovies = Xtrain.shape[1]

        # Random weights Initilization
        U = np.random.uniform(0, 1, (nrUsers, k))
        M = np.random.uniform(0, 1, (k, nrMovies))

        # Find Index of zero and non-zero elements for Train, Test
        (rows, cols, values) =  find(Xtrain)
        (nonRows, nonCols) = np.where(Xtrain==0)

        (rowsTest, colsTest, valuesTest) =  find(Xtest)
        (nonRowsTest, nonColsTest) = np.where(Xtest==0)

        errors = np.zeros((nrUsers, nrMovies))
        errorsTest = np.zeros((nrUsers, nrMovies))
        for i in range(numIter):
            for index,u in enumerate(U):
                movRated = np.where(Xtrain[index,:] != 0)[0]
                nrMovRated = len(movRated)
                ratUser = Xtrain[index, movRated]
                Ai = np.dot(M[:,movRated], M.T[movRated,:]) + penalty*nrMovRated*np.eye(k)
                Vi = np.dot(M[:,movRated], ratUser)
                res = np.dot(np.linalg.inv(Ai), Vi)
                U[index] = res

            for index,m in enumerate(M.T):
                usRated = np.where(Xtrain[:,index] != 0)[0]
                nrUsRated = len(usRated)
                ratMovie = Xtrain[usRated, index]
                if len(ratMovie)!=0:
                    Aj =  np.dot(U.T[:,usRated], U[usRated,:]) + penalty*nrUsRated*np.eye(k)
                    Vj = np.dot(U.T[:,usRated], ratMovie)

                    res = np.dot(np.linalg.inv(Aj), Vj)
                    M[:,index] =  res
                else:
                    M[:,index] = np.mean(M, axis=1)

            preds = np.dot(U, M)
            preds[np.where(preds > 5)] = 5
            preds[np.where(preds < 1)] = 1
            preds[nonRows, nonCols] = 0

            errors[rows, cols] = (preds - Xtrain)[rows, cols]
            RMSE[fold, i] = np.sqrt(np.sum(errors**2)/len(values))
            MAE[fold, i] = np.sum(np.abs(errors))/len(values)
            if i%10 == 0:
                print("Iteration is ", i, "and RMSE is ", RMSE[fold, i])
            if i != 0:
                if (RMSE[fold, i-1] - RMSE[fold, i]) < 0.000001:
                    print("Converged at iteration", i)
                    break
        timeFold[fold] = time.time() - beginTime
        print("fold " + str(fold)+" took "+ str(timeFold[fold]))

        errors = preds - Xtrain
        RMSEtrain[fold] = np.sqrt(np.sum(errors**2)/len(values))
        MAEtrain[fold] = np.sum(np.abs(errors))/len(values)

        preds = np.dot(U, M)
        preds[np.where(preds > 5)] = 5
        preds[np.where(preds < 1)] = 1
        preds[nonRowsTest, nonColsTest] = 0

        errorsTest[rowsTest, colsTest] = (preds - Xtest)[rowsTest, colsTest]

        RMSEtest[fold] = np.sqrt(np.sum(errorsTest**2)/len(valuesTest))
        MAEtest[fold] = np.sum(np.abs(errorsTest))/len(valuesTest)
        print("For fold ", fold, " RMSEtrain is ;", RMSEtrain[fold])
        print("For fold ", fold, " MAEtrain is ;", MAEtrain[fold])

        print("For fold ", fold, " RMSEtest is ;", RMSEtest[fold])
        print("For fold ", fold, " MAEtest is ;", MAEtest[fold], "\n")

        return(RMSEtrain, RMSEtest, MAEtrain, MAEtest)

data = loadData('ratings.dat')
matrixFactorizationALS(data, 2, 10, 1992, 40, 0.05)
