import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pouya as CT
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





"""arr = np.load("C:\PouyaResults/regularization_0.1batch_30fx1_1527266807.162152/Y_pred_processed.npy")
plt.plot(arr)  #hist
plt.show()"""

Y_new = np.load("C:\PouyaResults/regularization_0.01batch_30fx3_1527530768.9237857/Y_pred.npy")
print(X_pred.shape)
print(Y_new.shape)
plt.show()
plt.plot(Y_new[:,0])
plt.show()
plt.plot(Y_new[:, 1])
plt.show()
plt.plot(Y_new[:, 2])
plt.show()
plt.plot(Y_new[:, 3])
plt.show()
#new = CT.saveResults(arr, "C:/test.npy")




"""plt.plot(arr)
plt.show()"""
"""Y=arr
Y = Y[:,0,:,:]
Y_new = np.zeros(shape=(X_train.shape[0],20))
for idx, el in enumerate(Y):
    Y_new[idx] = el[0]
    if idx==X_train.shape[0]-30:
        for i in range(30):
            Y_new[idx+i] = el[i]
Y_new = Y_new + np.mean(Y_train)
plt.plot(Y_new)
plt.show()"""

"""arr = np.load("C:\Y_pred_processed.npy")
plt.plot(arr[:,0])
plt.show()
plt.plot(Y_train[:,0])
plt.show()"""