import scipy.io as sio
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from preprocessing import reject_outliers_and_standardize
data = sio.loadmat('C:/pouyafiles/fx2_LEON.mat')
#pprint(data)
#pprint(data)
X_train = data['X_train']
Y_train = data['Y_train']
X_pred = data['X_pred']
print(X_train.shape)
print(X_pred.shape)
print(Y_train.shape)

X_mean = np.mean(X_train)
X_std = np.std(X_train)
Y_mean = np.mean(Y_train)
Y_std = np.std(Y_train)


X_train = (X_train - np.mean(X_train)) / (np.std(X_train) + 0.000001)
Y_train = (Y_train - Y_mean) / (Y_std + 0.000001)
#plt.imshow(X_train[4])
#plt.show()

def saveResults(resultsDir, savedir, no_features = 8, size = 30):
    Y = np.load(resultsDir) #resultsDir C:/PouyaResults/Y_pred.npy'
    Y = Y[:,0,:,:]
    Y_new = np.zeros(shape=(X_train.shape[0],no_features))
    for idx, el in enumerate(Y):
        Y_new[idx] = el[0]
        if idx==X_train.shape[0]-size:
            for i in range(30):
                Y_new[idx+i] = el[i]

    #pprint(Y_new)
    Y_new = (Y_new )*Y_std + Y_mean

    pprint(Y_new.shape)
    plt.hist(Y_new)
    plt.show()
    plt.hist(data['Y_train'])
    plt.show()

    np.save(savedir, Y_new)



def getBatch(size = 30, maxsize = X_train.shape[0], minsize = 0):
    maxsize -=30
    rn = np.random.randint(0, maxsize+1-minsize, size = (size))
    arr = []
    coordinates = []
    for i in rn:
        arr.append(X_train[i])
        coordinates.append(Y_train[i])
    return np.array(arr), coordinates


def getValBatch(size = 30, maxsize = X_train.shape[0]):
    minsize = maxsize - size
    rn = np.arange(minsize,maxsize)
    arr = []
    coordinates = []
    for i in rn:
        arr.append(X_train[i])
        coordinates.append(Y_train[i])
    return np.array(arr), coordinates

