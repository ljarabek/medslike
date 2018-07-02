import scipy.io as sio
import numpy as np
from pprint import pprint
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import configparser
from preprocessing import reject_outliers_and_standardize

#pprint(data)
#pprint(data)

config = configparser.ConfigParser()
config.read('config.ini')
cfg = config['DEFAULT']
files_path = cfg['files_path']
BatchSize = int(cfg['batchsize'])
TrainMaxIndex = int(cfg['trainMaxIndex'])
testFrom = int(cfg['testFrom'])
testTo = int(cfg['testTo'])
path = cfg['MAT_path']

data = sio.loadmat(path)

#X_train = data['X_train']
#Y_train = data['Y_train']
#X_pred = data['X_pred']
#
#features = Y_train.shape[1]

#print(X_train.shape)
#print(X_pred.shape)
#print(Y_train.shape)

#X_mean = np.mean(X_train)
#X_std = np.std(X_train)
#Y_mean = np.mean(Y_train)
#Y_std = np.std(Y_train)


#X_train = (X_train - X_mean) / (np.std(X_train) + 0.000001)
#Y_train = (Y_train - Y_mean) / (Y_std + 0.000001)
#plt.imshow(X_train[4])
#plt.show()

"""def saveResults(results, savedir, no_features = features, size = BatchSize):
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
    
    return Y_new"""

def preprocess():
    data1 = sio.loadmat(files_path + 'fx1_XTRAIN.mat')
    data2 = sio.loadmat(files_path + 'Y_train.mat')

    X_train = data1['X_TRAIN']  # (110, 80, 80)

    Y_train = data2['Y_train']  # (110, 1024, 768, 1, 3)


    try:
        Y = np.load(files_path+"Y.npy")
        print("loaded Y from memory")
    except IOError:
        Z = np.zeros((110, 1024, 768, 2), dtype = np.float32)
        print("standardizing and saving Y")
        for idz, z in enumerate(Y_train):
            Z[idz,:,:,0] = np.transpose(np.transpose(z[:,:,0,0]) - np.arange(1024))
            Z[idz,:,:,1] = z[:,:,0,1] - np.arange(768)
        Y = Z/np.std(Z)
        del Z
        np.save(files_path+"Y.npy",Y)
    try:
        Y = np.load(files_path+"Y_se.npy")
        print("loaded SE from memory")
    except IOError:
        Y = np.reshape(Y, (Y.shape[0], Y.shape[1] * Y.shape[2] * Y.shape[3]))
        Y = SpectralEmbedding(6).fit_transform(X=Y)
        np.save(files_path+"Y_se.npy", Y)
        print("generated&saved SE")
    try:
        X = np.load(files_path+"X.npy")
        print("loaded X from memory")
    except IOError:
        print("standardizing and saving X")
        X = (X_train[:,23:39, 20:43] - np.mean(X_train[:,23:39, 20:43])) / np.std(X_train[:,23:39, 20:43])
        np.save(files_path + "X.npy", X)

    try:
        X_pred = np.load(files_path+"X_pred.npy")
        print("loaded X_pred from memory")
    except IOError:
        print("normalising and saving x_pred")
        X_pred = data1['X_PRED']
        X_pred = (X_pred[:,23:39, 20:43] - np.mean(X_train[:,23:39, 20:43])) / np.std(X_train[:,23:39, 20:43])
        np.save(files_path+"X_pred.npy",X_pred)

    return X, Y, X_pred #clear clutter from memory

X_train, Y_train, X_pred = preprocess()
#plt.imshow(X_train[50])
#plt.show()
#plt.hist(Y_train)
#plt.show()
def getBatch(size = BatchSize, maxsize = X_train.shape[0], minsize = 0):
    maxsize = maxsize - size
    rn = np.random.randint(0, maxsize+1-minsize, size = (size))
    arr = []
    coordinates = []
    for i in rn:
        arr.append(X_train[i])
        coordinates.append(Y_train[i])
    return np.array(arr), np.array(coordinates)


def getValBatch(size = BatchSize, maxsize = X_train.shape[0]):
    minsize = maxsize - size
    #rn = np.arange(minsize,maxsize)
    arr = []
    coordinates = []
    for i in np.arange(minsize,maxsize): # in rn
        arr.append(X_train[i])
        coordinates.append(Y_train[i])
    return np.array(arr), np.array(coordinates)

