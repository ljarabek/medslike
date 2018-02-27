import numpy as np
import CT
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import SimpleITK as sitk
import tensorflow as tf
from sklearn import preprocessing
import scipy.spatial as sps
import scipy.ndimage as spi
from tqdm import tqdm
from pprint import pprint

TrainSurfaceB, TrainCoordsB  = CT.getBatch(10, 250)
TestSurfaceB, TestCoordsB = CT.getBatchTest(10,251,299)

plt.imshow(TrainSurfaceB[2])
plt.show()







"""
#EXCLUDE outliers:
arr = np.load('C:/MEDSLIKE/numpy/surface.17/{}.npy'.format("vse"))

def reject_outliers_and_standardize(data, m=2):
    data[abs(data - np.mean(data)) > m * np.std(data)] = 0
    m = np.mean(data)
    s = np.std(data)
    data[data==0] = m
    for idx, i in tqdm(np.ndenumerate(data)): data[idx] = (i-m)/(s+0.000001)
    data[abs(data - np.mean(data)) > m * np.std(data)] = 0
    return data

nar = reject_outliers_and_standardize(arr, 2)
narr = np.array(nar)
print(narr.shape)
plt.imshow(narr[5])
plt.show()
#plt.imshow(reject_outliers(arr)[5])
#plt.show()





#print(np.load('C:/test.npy'))
#plt.plot(np.load('C:/test.npy'))
#plt.show()

#bla = np.load('C:/MEDSLIKE/numpy/surface.17/vse.npy')
#print(bla.shape)
"""
"""lol = np.load("C:/MEDSLIKE/numpy/surface.17/stddev.npy")
plt.imshow(lol)
plt.show()



for i in tqdm(range(299,590)):
    #CT.SurfaceAsCoordinates(i)
    yo = np.load('C:/MEDSLIKE/numpy/surface.17/{}.npy'.format(i))
    yo = CT.crop2DArr(yo,17,211,118,397)
    np.save('C:/MEDSLIKE/numpy/surface.17/{}.npy'.format(i), yo)"""
"""
arr, co = CT.getBatch(15)
plt.imshow(arr[12])
plt.show()
plt.plot(CT.standardizecoords(np.load('C:/MEDSLIKE/numpy/xyzTUMORJAzaPRVIH300slik.npy')))
plt.show()




plt.imshow(CT.getBatch()[2])
plt.show()

#povprečje, standardna deviacija in primeri, kako zgledajo standardizirani primeri (glej mapo 'fotke')
vse=np.load('D:/MEDSLIKE/numpy/surface.17/vse.npy')
povp = np.mean(vse, 0) #mean vsakega pixla
stdd = np.std(vse, 0) #stddev vsakega pixla
np.save('D:/MEDSLIKE/numpy/surface.17/povp.npy', povp)
np.save('D:/MEDSLIKE/numpy/surface.17/stddev.npy', stdd)
plt.imshow((povp-vse[6])/stdd)
plt.show()
plt.imshow((povp-vse[4])/stdd)
plt.show()
plt.imshow((povp-vse[2])/stdd)
plt.show()
"""



"""
for i in tqdm(range(300)):
    dr = CT.GetDir(i, 'D:/MEDSLIKE/numpy/surface.17/')
    vse.append(np.load(dr))
plt.imshow(vse[3])

plt.show()

np.save('D:/MEDSLIKE/numpy/surface.17/vse.npy', vse)


Tako smo naredili skene površja
for i in tqdm(range(300)):
    example = CT.SurfaceAsCoordinates(i)
    example = CT.crop2DArr(example,17,211,118,397)
    np.save('D:/MEDSLIKE/numpy/surface.17/{}.npy'.format(i), example)"""




""" 
SURFACE AS COORDINATES:
def SurfaceAsCoordinates(consecutive_number = 0, dirr='d:/MEDSLIKE', gradientThreshold = 0.17):
    img = CT.GetImage(consecutive_number,indir=dirr)
    imgG = sitk.GradientMagnitude(img)
    arrG = sitk.GetArrayFromImage(imgG)
    arrNew = np.array(np.zeros((len(arrG), len(arrG[0,0])))) #prva koordinata bo X, druga pa Z!!
    for z in tqdm(range(len(arrG))):
        for x in range(len(arrG[0,0])):
            for y in range(len(arrG[0])):
                if arrG[z,y,x]>gradientThreshold:
                    arrNew[z,x] = y
                    break
    plt.imshow(arrNew)
    plt.show()
    return arrNew
SurfaceAsCoordinates(5)
#print(SurfaceAsCoordinates(5)[85,140,286]) #ARRAY JE Z - Y - X
"""

"""
haha = np.load('D:/MEDSLIKE/outputs/{}.npy'.format(4))
haha = haha.squeeze(0)
haha = haha.squeeze(3)
maxindex = [0,0,0]
maxvalue = 0
for i in range(len(haha)):
    for j in range(len(haha[0])):
        for k in range(len(haha[0,0])):
            if haha[i,j,k] > maxvalue:
                maxvalue = haha[i,j,k]
                maxindex = [i,j,k]

print('Maxvalue {0} at {1}'.format(maxvalue,maxindex))
"""

"""(redundantno*)
GET MAX INDEX**
def GetMaxIndex(c, directory='D:/MEDSLIKE/outputs/{}.npy'):  # vrne index maximuma (redundantno*)
    haha = np.load('D:/MEDSLIKE/outputs/{}.npy'.format(0))
    bla = np.zeros([c, 3])
    for g in range(c):
        haha = np.load('D:/MEDSLIKE/outputs/{}.npy'.format(g))
        haha = haha.squeeze(0)
        haha = haha.squeeze(3)
        maxvalue = 0
        x = len(haha[0, 0])
        y = len(haha[0])
        z = len(haha)
        for i in range(z):
            for j in range(y):
                for k in range(x):
                    if haha[i, j, k] > maxvalue:
                        maxvalue = haha[i, j, k]
                        bla[g] = [i, j, k]
    return bla"""

""""#tole poišče vsa težišča in jih shrani
def GetMaxWeightedIndex(c, directory='C:/MEDSLIKE/outputsNEWall/{}.npy'):
    # vrne tuple xyz z koordinatami težišča slike; ista funkcija je v CT
    haha = np.load(directory.format(c))  # prebere slike z verjetnosti/'logits' tumorja na lokacijah (output ročno-nastavljenega CNN-ja)
    #haha = haha.squeeze(0)  # zarad tensorflowa je prejšni array oblike 1,x,y,z,1 --> squeeze da dobimo xyz
    #haha = haha.squeeze(3)
    return spi.center_of_mass(haha)
my_array = []
for i in tqdm(range(0,500)):
    my_array.append(GetMaxWeightedIndex(i))
np.save('C:/MEDSLIKE/XYZ/train500.npy', my_array)
print(np.array(my_array).shape)
my_array = []
for i in tqdm(range(501,590)):
    my_array.append(GetMaxWeightedIndex(i))
np.save('C:/MEDSLIKE/XYZ/test89.npy', my_array)
print(np.array(my_array).shape)"""

"""
my_arrayx = []
my_arrayy = []
my_arrayz = []
for c in tqdm(range(300)):
    center_of_mass = GetMaxWeightedIndex(c)
    my_arrayx.append(center_of_mass[0])
    my_arrayy.append(center_of_mass[1])
    my_arrayz.append(center_of_mass[2])             #naredi posebej 3 arraye z x, y, z koordinatami:

my_arrayxyz = [[0 for i in range(3)] for j in range(300)]
#print(len(my_arrayxyz))     300 OBSERVATIONS
#print(len(my_arrayxyz[0]))  3 FEATURES (koordinate)
for i in range(len(my_arrayxyz)):
    my_arrayxyz[i][0] = my_arrayx[i]
    my_arrayxyz[i][1] = my_arrayy[i]
    my_arrayxyz[i][2] = my_arrayz[i]                #tle te 3 arraye spnemo v 1 array xyz

np.save("D:/MEDSLIKE/numpy/xyzTUMORJAzaPRVIH300slik.npy", my_arrayxyz)
print(my_arrayxyz)                                  #array se shrani, output(vrstice so primeri=vektorji, stolpci pa koordinate): 
                                                    #[[14.187024436022119, 15.010320902770035, 10.978855061772665], 
                                                    #[14.14542350136853, 15.009811974257069, 10.946614117253652], 
                                                    #[14.060729379700343, 14.983834120329774, 10.896912115915894], ..."""


""" TOLE NARIŠE GRAF (kot na sliki, le da so še XY XZ YZ grafi gor(sklenjeni*):
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
ax1.plot(my_arrayx)
ax1.set_title("x - y - z")
ax2.plot(my_arrayy)
ax3.plot(my_arrayz)
ax4.plot(my_arrayx, my_arrayy)
ax4.set_title("x - y, x-z, y-z")
ax5.plot(my_arrayx,my_arrayz)
ax6.plot(my_arrayy,my_arrayz)
plt.show()

"""

""" tole je krneki
base = sitk.ReadImage('C:/00000.mha')
slicexx = 104
for i in tqdm(range(250)):
    B  = CT.getImage(i)
    Barr = CT.LoadCTAsArray(i)
    for x in range(len(Barr[slicexx])): #zbrali smo si slice slicexxx (116 prej, zdej 104)
        for y in range(len(Barr[slicexx,x])):
            base[y,x,i] = np.float32(Barr[slicexx, x, y]).item()
#^^DELA!! sitk.Show(base)
#print(base[1,1,1].dtype)
#print(np.dtype)
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict = {input1: [7.3], input2:[8.]})) #https://youtu.be/6rDWwL6irG0?t=224 !!!

def neural_network_model(X):
    X = tf.placeholder(tf.float32)
    w = tf.placeholder(tf.float32)
    output = tf.multiply(X, w)
    return output

y=[2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,4,5,]
def train_neural_network(X):
    prediction = neural_network_model(X)
    cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs = 10
    
    
"""
