import numpy as np
import tensorflow as tf
import CT as CT
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
cfgt = config['TUMOR BOUNDARIES']
cfgr = config['REGION BOUNDARIES']
cfg = config['DEFAULT']
BatchSize = int(cfg['batchSize'])
TrainMaxIndex = int(cfg['trainMaxIndex'])
testFrom = int(cfg['testFrom'])
testTo = int(cfg['testTo'])
regularization = float(cfg['regularization'])
path = cfg['MHA_path']


"""CROPS TUMOR"""
hah = CT.GetImage(10)
x1 = int(cfgt['x1']) #Exact tumor boundaries
x2 = int(cfgt['x2'])
y1 = int(cfgt['y1'])
y2 = int(cfgt['y2'])
z1 = int(cfgt['z1'])
z2 = int(cfgt['z2'])
tumor = {'x':x2-x1 , 'y': y2-y1 , 'z':z2-z1}
kernel = CT.CropImage(x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2,image = hah) #GET kernelIMAGE = tumor
kernel = sitk.GetArrayFromImage(kernel)
kernel = CT.ConvertArrayToBinary(kernel, threshold=0.5) #converts the tumor to binary image
kernel = kernel.reshape([tumor['z'], tumor['y'], tumor['x'], 1, 1]) #adjust shape for feeding into tensorflow

"""CROPS GENERAL AREA OF THE TUMOR (to reduce time) + overwrites tumor boundaries (kernel was already obtained)"""

x1 = int(cfgr['x1'])
x2 = int(cfgr['x2'])
y1 = int(cfgr['y1'])
y2 = int(cfgr['y2'])
z1 = int(cfgr['z1'])
z2 = int(cfgr['z2'])
area_dimension = {'x':x2-x1 , 'y': y2-y1 , 'z':z2-z1}


def getArea(No, x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2):
    imag = CT.GetImage(No)
    ima = CT.CropImage(x1,x2,y1,y2,z1,z2, imag)
    area = sitk.GetArrayFromImage(ima)
    area = area.reshape([1, area_dimension['z'], area_dimension['y'], area_dimension['x'], 1])
    return area

strides = 1

w = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape = [1, area_dimension['z'], area_dimension['y'], area_dimension['x'], 1])
y = tf.nn.conv3d(x, w, (1, strides, strides, strides, 1), padding = 'VALID') #padding must be set to VALID
y = tf.nn.relu(y)

if not os.path.exists(path + 'FeatureMaps/'):
    os.makedirs(path + 'FeatureMaps/')


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    output = np.zeros(shape = (1,23,27,25,1))
    for i in tqdm(range(590)):
        convoluted = sess.run(y, feed_dict={w:kernel, x:getArea(i, x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)})
        output = convoluted
        output = output.squeeze(0)
        output = output.squeeze(3)
        np.save(path + 'FeatureMaps/{}.npy'.format(i), output)

#np.save('c:/balbla.npy', convoluted)
arr=[]
for i in tqdm(range(testTo)): #only 550 scans were used***
    arr.append(CT.GetMaxWeightedIndex(i,path + 'FeatureMaps/{}.npy'))
np.save(path + 'FeatureMaps/all.npy', arr=arr) #SAVES RELATIVE tumor coordinates as array (550,3) (relative to cropped general area)
plt.plot(arr)
plt.show()

"""print("output shape : {}".format(output.shape))
output = convoluted
output = output.squeeze(0)
output = output.squeeze(3)"""

"""np.save('c:/balbla.npy', output)
CT.ShowArrayAsCT(output, Name='output')
area = GetArea(0)
area = area.squeeze(axis = (0,4))
CT.ShowArrayAsCT(area, Name = 'input')"""


"""
#koordinate področja, ki ga gledamo:
x1 = 160
x2 = 201 #41 (+20)
y1 = 220
y2 = 261 #41 (+20)
z1 = 100
z2 = 131 #31 (+15)
area = CT.CropImage(x1,x2,y1,y2,z1,z2, hah)
sitk.Show(area, title = 'input')
area_dimension = {'x':x2-x1 , 'y': y2-y1 , 'z':z2-z1}
area = sitk.GetArrayFromImage(area)
area = area.reshape([1, area_dimension['z'], area_dimension['y'], area_dimension['x'], 1])
"""

"""
print("output shape : {}".format(output.shape))
print('input shape: {}'.format(area.shape))
print('kernel shape: {}'.format(kernel.shape))
output shape : (23, 27, 25)
input shape: (1, 31, 41, 41, 1)
kernel shape: (9, 15, 17, 1, 1)
"""