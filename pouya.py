import scipy.io as sio
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import configparser
from preprocessing import reject_outliers_and_standardize

#pprint(data)
#pprint(data)

config = configparser.ConfigParser()
config.read('config.ini')
cfg = config['DEFAULT']
BatchSize = int(cfg['batchsize'])
TrainMaxIndex = int(cfg['trainMaxIndex'])
testFrom = int(cfg['testFrom'])
testTo = int(cfg['testTo'])
path = cfg['MAT_path']

data = sio.loadmat(path)

X_train = data['X_train']
Y_train = data['Y_train']
X_pred = data['X_pred']

features = Y_train.shape[1]

#print(X_train.shape)
#print(X_pred.shape)
#print(Y_train.shape)

X_mean = np.mean(X_train)
X_std = np.std(X_train)
Y_mean = np.mean(Y_train)
Y_std = np.std(Y_train)


X_train = (X_train - X_mean) / (np.std(X_train) + 0.000001)
Y_train = (Y_train - Y_mean) / (Y_std + 0.000001)
#plt.imshow(X_train[4])
#plt.show()

def saveResults(results, savedir, no_features = features, size = BatchSize):
    #Y = np.load(resultsDir) #resultsDir C:/PouyaResults/Y_pred.npy'
    Y = results[:,0,:,:]
    Y_new = np.zeros(shape=(X_pred.shape[0],no_features))
    for idx, el in enumerate(Y):
        Y_new[idx] = el[0]
        if idx==X_pred.shape[0]-size:
            for i in range(size):
                Y_new[idx+i] = el[i]

    #pprint(Y_new)
    Y_new = (Y_new )*Y_std + Y_mean

    np.save(savedir, Y_new)
    pprint(Y_new.shape)
    #plt.plot(Y_new[:,3])
    #plt.show()
    """plt.plot(data['Y_train'])
    plt.show()
    plt.plot(Y_new[:,0])
    plt.show()
    plt.plot(Y_new[:, 1])
    plt.show()
    plt.plot(Y_new[:, 2])
    plt.show()
    plt.plot(Y_new[:, 3])
    plt.show()"""
    return Y_new



def getBatch(size = BatchSize, maxsize = X_train.shape[0], minsize = 0):
    maxsize = maxsize - size
    rn = np.random.randint(0, maxsize+1-minsize, size = (size))
    arr = []
    coordinates = []
    for i in rn:
        arr.append(X_train[i])
        coordinates.append(Y_train[i])
    return np.array(arr), coordinates


def getValBatch(size = BatchSize, maxsize = X_train.shape[0]):
    minsize = maxsize - size
    rn = np.arange(minsize,maxsize)
    arr = []
    coordinates = []
    for i in rn:
        arr.append(X_train[i])
        coordinates.append(Y_train[i])
    return np.array(arr), coordinates

