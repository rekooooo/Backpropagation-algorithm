import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def intialize(n_h, L, epochs, bias, activationFunction, nn):


    print(n_h, L, epochs, bias, activationFunction, nn)

    nn_s = re.findall(r"[\w']+", nn)
    NoOfNeurons = list(map(int, nn_s))
    #print(NoOfNeurons)

    AllData = pd.read_csv("iris.csv", header=None)
    #print(AllData)
    X = AllData.iloc[:, 0:4].values
    Y = AllData.iloc[:, -1].values

    #take 30 from every class

    class1X = X[0:50, :]
    class2X = X[50:100, :]
    class3X = X[100:150, :]

    class1Y = Y[0:50]
    class2Y = Y[50:100]
    class3Y = Y[100:150]

    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(class1X, class1Y, test_size=0.4)
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(class2X, class2Y, test_size=0.4)
    X_train3, X_test3, Y_train3, Y_test3 = train_test_split(class3X, class3Y, test_size=0.4)

    X_train = np.concatenate((X_train1,X_train2, X_train3))
    Y_train = np.concatenate((Y_train1,Y_train2, Y_train3))


    X_train, Y_train = shuffle(X_train, Y_train)


    X_test = np.concatenate((X_test1,X_test2, X_test3))
    Y_test = np.concatenate((Y_test1,Y_test2, Y_test3))


    X_test, Y_test = shuffle(X_test, Y_test)


    Y_train= np.reshape(Y_train, (Y_train.shape[0], 1))
    Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))

    Y_train_Matrix = np.zeros((90, 3))
    for i in range(Y_train_Matrix.shape[0]):

        if Y_train[i] == "Setosa":
            Y_train_Matrix[i] = np.array([1,0,0])
        elif Y_train[i] == "Versicolor":
            Y_train_Matrix[i] = [0, 1, 0]
        else:
            Y_train_Matrix[i] = [0, 0, 1]


    Y_test_Matrix = np.zeros((60, 3))
    for i in range(Y_test_Matrix.shape[0]):

        if Y_test[i] == "Setosa":
            Y_test_Matrix[i] = np.array([1,0,0])
        elif Y_test[i] == "Versicolor":
            Y_test_Matrix[i] = [0, 1, 0]
        else:
            Y_test_Matrix[i] = [0, 0, 1]

    W, b = train(X_train, Y_train_Matrix, n_h, L, epochs, bias, activationFunction, NoOfNeurons)
    return test(X_test, Y_test_Matrix, W, b, activationFunction)


def intialize_Parameters(N_X,nn, nClasses):
    nn.insert(0, N_X)
    nn.insert(len(nn), nClasses)

    W = []
    b = []
    i = 0
    for i in range(1, len(nn)):
        W.append(np.random.randn(nn[i], nn[i-1])*0.01)
        #W.append(np.zeros((nn[i], nn[i-1])))
        b.append(np.zeros((nn[i],1)))

    return W, b


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

def activation(A, W, b, activation):

    Z = np.dot(W,A)+b
    #print(W.shape, ".", A.shape, "+", b.shape,"=",Z.shape)
    if activation == "sigmoid":
        sig = sigmoid(Z)
        return Z,sig
    else:
        tan = tanh(Z)
        return Z,tan

def forward(X,W,b, activationFunction):

    A = X
    caches = []
    for i in range(len(W)):

        PrevA = A

        Z,A = activation(PrevA, W[i], b[i], activationFunction)

        cache = (Z, A)
        caches.append(cache)

    return A, caches




def sigmoid_derv(Z):
    A = sigmoid(Z)
    return A*(1-A)

def tanh_derv(Z):
    A = tanh(Z)
    return (1-(A**2))


def backward(W, y, AL, caches, activationFunction):

    Z, A = caches[-1]

    error = []

    if activationFunction == "sigmoid":
        error.append((y - AL) * sigmoid_derv(Z))
    else:
        error.append((y - AL) * tanh_derv(Z))

    j = 0
    res = 0
    for i in reversed(range(len(W))):

        if i == 0:
            break
        currentW = W[i]
        PrevError = error[j]
        Z, A = caches[i-1]

        if activationFunction=="sigmoid":
            res = sigmoid_derv(Z)
        else:
            res = tanh_derv(Z)

        tmp = np.dot(currentW.T, PrevError)

        out = np.multiply(tmp , res)

        error.append(out)
        #print (currentW.shape, "*", PrevError.shape, "*", sig.shape)

        j += 1

    return error



def update_weights(x, W, b, caches, L, error, activationFunction):


    #update the biss as well f0=1 or x0=1

    newWeights = []
    newBias = []
    error.reverse()
    currentW = W[0]
    currentE = error[0]
    currentF = x
    currentBias = b[0]
    currentBiasF = np.ones((error[0].shape[0], 1))

    newW = currentW + (L * np.dot(currentE, currentF.T))
    newWeights.append(newW)


    newB = currentBias + (L * np.multiply(currentE, currentBiasF))
    newBias.append(newB)

    #print(currentBias.shape, "+", currentE.shape,"*", currentBiasF.shape)
    #print(newB.shape)
    for i in range(1, len(W)):
        Z, A = caches[i-1]

        currentW = W[i]
        currentE = error[i]

        if activationFunction == "sigmoid":
            currentF = sigmoid(Z)
        else:
            currentF = tanh(Z)

        newW = currentW + L * np.dot(currentE, currentF.T)
        newWeights.append(newW)

        #bias update

        currentBias = b[i]
        currentBiasF = np.ones((error[i].shape[0], 1))

        newB = currentBias + (L * np.multiply(currentE, currentBiasF))
        newBias.append(newB)

        #print(currentBias.shape, "+", currentE.shape, "*", currentBiasF.shape)
        #print(newB.shape)

    #print("\n")


    return newWeights, newBias


def train(X, Y, n_h, L, epochs, bias, activationFunction, nn):

    W,b = intialize_Parameters(X.shape[1],nn, 3)
    #print("W ",W)
    #print("b ",b,"\n")
    for i in range(epochs):
        sum = 0
        for j in range(X.shape[0]):

            x = X[j].T
            x = x.reshape((4,1))

            y = Y[j].T
            y = y.reshape((3,1))

            AL, caches = forward(x, W, b, activationFunction)

            e = np.sum((y - AL)**2)
            sum+=e

            error = backward(W, y, AL, caches, activationFunction)
            newW, newB = update_weights(x, W, b, caches, L, error, activationFunction)
            W = newW

            if bias!=0:
                b = newB


        #print(sum,"\n")


    #print("W ",W)
    #print("b ",b)

    return W, b



def test(X, Y, W, b, activationFunction):

    accuracy = 0
    predicted = []
    YValue = []
    for i in range(X.shape[0]):
        x = X[i].T
        x = x.reshape((4, 1))

        y = Y[i].T
        y = y.reshape((3, 1))

        AL, caches = forward(x, W, b, activationFunction)

        max = np.max(AL)

        if max == AL[0]:
            predicted.append("Setosa")
        elif max == AL[1]:
            predicted.append("Versicolor")
        else:
            predicted.append("Virginica")


        if y[0]==1 and y[1]==0 and y[2]==0:
            YValue.append("Setosa")
        elif y[0]==0 and y[1]==1 and y[2]==0:
            YValue.append("Versicolor")
        else:
            YValue.append("Virginica")


        if predicted[i] == YValue[i]:
            accuracy+=1


    results = confusion_matrix(YValue, predicted, labels=["Setosa", "Versicolor", "Virginica"])
    #print(results)

    #accScore = accuracy_score(YValue, predicted)
    #print(accScore)

    #print("accuracy is: ", (accuracy/X.shape[0])*100)
    accuracy = (accuracy/X.shape[0])*100
    return results, accuracy


#intialize(3,0.1,2000,1,"sigmoid", "7")