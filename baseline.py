import numpy as np
import time as t


# Loading the data using only 3 columns as integers
def loadData(filename):
    """
    params:
        filename: the path to the data file (ratings.dat)
    return:
        np array of the data that was being read
    """
    data = np.genfromtxt(filename, usecols=(0, 1, 2), delimiter='::', dtype='int')
    return(data)

def RMSE(predicted, truth):
    """
    params:
        predicted: an array of the predicted ratings
        truth: the actual ratings in the data
    returns:
        the RMSE value
    """
    error = np.sqrt(np.mean((predicted - truth)**2))
    return(error)

def MAE (predicted, truth):
    """
    params:
        predicted: an array of the predicted ratings
        truth: the actual ratings in the data
    returns:
        the MAE value
    """
    error = np.mean(np.abs(predicted - truth))
    return(error)

def calculatePrediction(data, predictionBy):
    """
    params:
        data: either train or test
        predictionBy: either user or item which is supplied in the parent function
    returns:
        calculated predictions per user or item
    """
    if predictionBy == "user":
        # Initalize predictions array to be filled
        predictions = np.zeros(np.max(data[:,0]))

        # Get frequency and range using bincount and cumsum
        bins = np.bincount(data[:,0])
        cumSum = np.cumsum(bins)

        # Calculate the mean rating of each user
        for i in range(np.max(data[:,0])):
            temp = data[cumSum[i]:cumSum[i+1],2]
            if len(temp)>1:
                predictions[i]  = np.mean(temp)
            else:
                predictions[i] = np.mean(data[:,2])

        result = np.repeat(predictions, bins[1:])

    elif predictionBy == "item":
        # Initalize predictions array to be filled
        predictions = np.zeros(np.max(data[:,1]))

        # Get frequency and range using bincount and cumsum
        bins = np.bincount(data[:,1])
        cumSum = np.cumsum(bins)

        # Calculate the mean rating of each item
        for i in range(np.max(data[:,1])):
            temp = data[cumSum[i]:cumSum[i+1],2]
            if len(temp)>1:
                predictions[i]  = np.mean(temp)
            else:
                predictions[i] = np.mean(data[:,2])
        result = np.repeat(predictions, bins[1:])

    return(result)

def getPredictions(train, test, predictionBy):
    """
    params:
        train: the train data
        test: the test data
        predictionBy: either "user" or "item"

    returns:
        the predictions for train and test based on user preference user or item
    """

    if predictionBy == "user":
        predictionsTrainUser = calculatePrediction(train, predictionBy)
        predictionsTestUser = calculatePrediction(test, predictionBy)

        return(predictionsTrainUser, predictionsTestUser)

    elif predictionBy == "item":
        predictionsTrainItem = calculatePrediction(train, predictionBy)
        predictionsTestItem = calculatePrediction(test, predictionBy)

        return(predictionsTrainItem, predictionsTestItem)
    else:
        print("wrong ... use 'user' or 'item'")


def globalAverageModel(train, test):
    """
    params:
        train: the train data
        test: the test data
    returns:
        arrays of the RMSE and MAE values on the train and test sets
    """
    predictionsTrain = np.mean(train[:,2])
    predictionsTest = np.mean(test[:,2])

    errorTrainRMSE = RMSE(predictionsTrain, train[:,2])
    errorTestRMSE = RMSE(predictionsTest, test[:,2])

    errorTrainMAE = MAE(predictionsTrain, train[:,2])
    errorTestMAE = MAE(predictionsTest, test[:,2])

    return(errorTrainRMSE, errorTestRMSE, errorTrainMAE,  errorTestMAE)


def userAverageModel(train, test):
    """
    params:
        train: the train data
        test: the test data
    returns:
        arrays of the RMSE and MAE values on the train and test sets
    """
    indexTrain = train[:,0].argsort()
    train = train[indexTrain]

    indexTest = test[:,0].argsort()
    test = test[indexTest]

    predictionsTrain, predictionsTest = getPredictions(train, test, "user")

    errorTrainRMSE = RMSE(predictionsTrain, train[:,2])
    errorTestRMSE = RMSE(predictionsTest, test[:,2])

    errorTrainMAE = MAE(predictionsTrain, train[:,2])
    errorTestMAE = MAE(predictionsTest, test[:,2])

    return(errorTrainRMSE, errorTestRMSE, errorTrainMAE,  errorTestMAE)



def itemAverageModel(train, test):
    """
    params:
        train: the train data
        test: the test data
    returns:
        arrays of the RMSE and MAE values on the train and test sets
    """
    indexTrain = train[:,1].argsort()
    train = train[indexTrain]

    indexTest = test[:,1].argsort()
    test = test[indexTest]

    predictionsTrain, predictionsTest = getPredictions(train, test, "item")

    errorTrainRMSE = RMSE(predictionsTrain, train[:,2])
    errorTestRMSE = RMSE(predictionsTest, test[:,2])

    errorTrainMAE = MAE(predictionsTrain, train[:,2])
    errorTestMAE = MAE(predictionsTest, test[:,2])

    return(errorTrainRMSE, errorTestRMSE, errorTrainMAE,  errorTestMAE)

def regressionModel(train, test):
    """
    params:
        train: the train data
        test: the test data
    returns:
        arrays of the RMSE and MAE values on the train and test sets
    """
    # Preprocessing
    indexTrain = train[:,0].argsort()
    train = train[indexTrain]

    indexTest = test[:,0].argsort()
    test = test[indexTest]

    predictionsTrainUser, predictionsTestUser = getPredictions(train, test, "user")

    indexTrain = train[:,1].argsort()
    train = train[indexTrain]

    indexTest = test[:,1].argsort()
    test = test[indexTest]

    predictionsTrainItem, predictionsTestItem = getPredictions(train, test, "item")

    predictionsTrain, yTrain = getRegressionPredictions(train, predictionsTrainUser, predictionsTrainItem)
    predictionsTest, yTest = getRegressionPredictions(test, predictionsTestUser, predictionsTestItem)

    errorTrainRMSE = RMSE(predictionsTrain, yTrain)
    errorTestRMSE = RMSE(predictionsTest, yTest)

    errorTrainMAE = MAE(predictionsTrain, yTrain)
    errorTestMAE = MAE(predictionsTest, yTest)

    return(errorTrainRMSE, errorTestRMSE, errorTrainMAE,  errorTestMAE)

def getRegressionPredictions(data, userPredictions, itemPredictions):
    """
    params:
        data: the train or test data
        userPredictions: predictions of average rating per user
        itemPredictions: predictions of average rating per item
    returns:
        prediction of regression along with the true labels "y"
    """
    # data can be train or test and the predictions of user of item should match the data
    data = np.column_stack((data, itemPredictions))

    data = data[np.lexsort((data[:,1], data[:,0]))]

    interceptData = np.repeat(1, userPredictions.shape)

    X = np.vstack((interceptData, userPredictions, data[:,3])).T

    y = data[:,2]

    coefs = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    predicted = np.dot(X, coefs)

    # if a prediction is less than 1 we set it to 1 and if greater than 5 we set it to 5
    predicted[np.where(predicted < 1)] = 1
    predicted[np.where(predicted > 5)] = 5

    return(predicted, y)

def main(data, nfolds, model, seed):
    """
    params:
        data: the full dataset of ratings
        nfolds: number of folds for cross-validation
        model: the name of the naive model (user, item, global or regression)
        seed: random seed value
    """
    if model == "user" or model == "item" or model == "global" or model == "regression":
        # Define some variables that will be used later
        # Errors Array for RMSE
        errorTrainRMSE = np.zeros(nfolds)
        errorTestRMSE = np.zeros(nfolds)
        # Errors Array for MAE
        errorTrainMAE = np.zeros(nfolds)
        errorTestMAE = np.zeros(nfolds)

        # Set random seed
        np.random.seed(seed)

        # Create a sequence and shuffle it to split the data to train and test
        seqs = [x % nfolds for x in range(len(data))]
        np.random.shuffle(seqs)
        # Begin cross validation loop
        for fold in range(nfolds):
            # Define indexes to select both train and test
            trainSelect = np.array([x != fold for x in seqs])
            testSelect = np.array([x == fold for x in seqs])

            # Select train and test
            train = data[trainSelect]
            test = data[testSelect]

            if model == "user":
                errorTrainRMSE[fold], errorTestRMSE[fold], errorTrainMAE[fold], errorTestMAE[fold] = userAverageModel(train, test)
            elif model == "item":
                errorTrainRMSE[fold], errorTestRMSE[fold], errorTrainMAE[fold], errorTestMAE[fold] = itemAverageModel(train, test)
            elif model == "global":
                errorTrainRMSE[fold], errorTestRMSE[fold], errorTrainMAE[fold], errorTestMAE[fold] = globalAverageModel(train, test)
            else:
                errorTrainRMSE[fold], errorTestRMSE[fold], errorTrainMAE[fold], errorTestMAE[fold] = regressionModel(train, test)

        print("Average RMSE_train over the folds =" + str(np.mean(errorTrainRMSE)))
        print("Average RMSE_test over the folds =" + str(np.mean(errorTestRMSE)))
        print("Average MAE_train over the folds =" + str(np.mean(errorTrainMAE)))
        print("Average MAE_test over the folds =" + str(np.mean(errorTestMAE)))
    else:
        print("please type in a valid model. You may choose user or item")



loadDataTime = t.time()
data = loadData("ratings.dat")
print("Loading the data took ", t.time() - loadDataTime,'\n')

print("global now",'\n')

globalTime = t.time()
main(data, 5, "global", 17)
print("Global average took ", t.time() - globalTime,'\n')

print("item now",'\n')

itemTime = t.time()
main(data, 5, "item", 17)
print("Item average took ", t.time() - itemTime,'\n')

print("user now",'\n')

userTime = t.time()
main(data, 5, "user", 17)
print("User average took ", t.time() - userTime,'\n')

print("regression",'\n')

regTime = t.time()
main(data, 5, "regression", 17)
print("regression took", t.time() - regTime,'\n')
