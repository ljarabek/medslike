import numpy as np
import CT
from preprocessing import reject_outliers_and_standardize
import os
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import SimpleITK as sitk
import tensorflow as tf
from sklearn import preprocessing
import scipy.spatial as sps
import scipy.ndimage as spi
from tqdm import tqdm
from pprint import pprint
#from preprocessing import reject_outliers_and_standardize_coords
config = configparser.ConfigParser()

config.read('config.ini')
cfg = config['DEFAULT']
path = cfg['MHA_path']
print(path+'outputs/hha')
"""
#PLOT TEST-TRAIN LOSS
costs = np.load('C:/MEDSLIKE/RESULTS/every50/costs.npy')
costs = costs.transpose()
print(costs.shape)
print(costs[0]) #train
print(costs[1]) #test
#plt.plot(range(45740), costs[0][9:], 'r--', range(45750), costs[1], 'b--')
plt.semilogy(range(45741), costs[0][9:], 'r--', range(45741), costs[1][9:], 'b--')  # y  is logarithmic
plt.show()"""


"""vse=np.load('C:/MEDSLIKE/numpy/surface.17/vse.npy')
plt.imshow(vse[5])
plt.show()
kvse = np.load('C:/MEDSLIKE/outputsNEWall/vsiXYZ.npy')
results = np.load('C:/MEDSLIKE/RESULTS/every50/testreg05.npy') #0infer-1answer-2cost
kstdd = np.nanstd(kvse,0)
results[0] *= kstdd
results[1] *= kstdd
plt.plot(range(49), results[0], 'r--', range(49), results[1], 'b--')
plt.show()
kvse = np.load('C:/MEDSLIKE/outputsNEWall/vsiXYZ.npy') #koordinate
kpovp = np.nanmean(kvse,0)
kstdd = np.nanstd(kvse,0)
pprint(kpovp)
results = np.load('C:/MEDSLIKE/RESULTS/every50/testreg05.npy') #0infer-1answer-2cost
results[0] *= kstdd
results[1] *= kstdd
#results = np.reshape(results, (3,-1))
pprint('COST {}'.format(results[2]))
pprint(np.mean(results[0]-results[1],0))
pprint(np.std(abs(results[0]-results[1]),0))

plt.plot(range(49), results[0], 'r--', range(49), results[1], 'b--')
#pprint((sps.distance.euclidean(results[0][0],results[1][0])))
a=[]
for id, i in enumerate(results[0]): a.append(sps.distance.euclidean(i, results[1][id]))
pprint(a)
pprint(np.mean(a))
pprint(np.std(a))
plt.plot(results[0]-results[1])
plt.show()"""
#pprint(results[1])
#pprint("Bla {}".format(results[1,1]))

"""hah = np.load('C:/MEDSLIKE/RESULTS/1.0/trainreg05.npy')
hah = np.reshape(hah, (3,-1))
print(hah.shape)
pprint(hah[:,2])
bla, k = CT.getBatch(60)
print(k[5])
plt.imshow(bla[5])
plt.show()
print(k[25])
plt.imshow(bla[25])
plt.show()

kor = np.load('C:/MEDSLIKE/outputsNEWall/vsiXYZ.npy')
#pic, ko = CT.getBatch()
bla, koord = CT.getBatch()

print(koord)
print(np.array(koord).shape)

bla, koord = CT.getBatchTest()

print(koord)
print(np.array(koord).shape)

#korr = [list(elem) for elem in kor]
#pprint(kor)
korpovp = np.nanmean(kor,0)

korstd = np.nanstd(kor,0)
print(korpovp, korstd)
print((kor-korpovp)/korstd)
#CT.ShowArrayAsCT(np.load('C:/MEDSLIKE/outputsNEWall/10.npy'), Name='10')
#.ShowArrayAsCT(np.load('C:/MEDSLIKE/outputsNEWall/20.npy'), Name='20')"""

"""arr=[]
for i in tqdm(range(550)):
    arr.append(CT.GetMaxWeightedIndex(i,'C:/MEDSLIKE/outputsNEWall/{}.npy'))
np.save('C:/MEDSLIKE/XYZ/vsiXYZ.npy', arr=arr)"""
"""vse=np.load('C:/MEDSLIKE/numpy/surface.17/vse.npy')
kvse = np.load('C:/MEDSLIKE/XYZ/vsiXYZ.npy')
print(kvse)"""



"""bla, k = CT.getBatch()
print(k)
plt.imshow(bla[5])
plt.show()
plt.imshow(bla[25])
plt.show()"""

"""vse = []


for i in range(550):
    vse.append(np.load('C:/MEDSLIKE/numpy/surface.17/{}.npy'.format(i)))
print(np.array(vse).shape)
plt.imshow(vse[3])

plt.show()

np.save('C:/MEDSLIKE/numpy/surface.17/vse.npy', vse)"""


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




""""""



"""for i in tqdm(range(350)):
    example = CT.SurfaceAsCoordinates(i)
    print(i)
    #example = CT.crop2DArr(example,17,211,118,397)
    np.save('C:/MEDSLIKE/numpy/surface.17/{}.npy'.format(i), example)"""




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

#tole poišče vsa težišča in jih shrani
"""def GetMaxWeightedIndex(c, directory='C:/MEDSLIKE/outputsNEWall/{}.npy'):
    # vrne tuple xyz z koordinatami težišča slike; ista funkcija je v CT
    haha = np.load(directory.format(c))  # prebere slike z verjetnosti/'logits' tumorja na lokacijah (output ročno-nastavljenega CNN-ja)
    #haha = haha.squeeze(0)  # zarad tensorflowa je prejšni array oblike 1,x,y,z,1 --> squeeze da dobimo xyz
    #haha = haha.squeeze(3)
    return spi.center_of_mass(haha)
my_array = []
for i in range(590):
    my_array.append(GetMaxWeightedIndex(i))
    print(my_array)
    np.save('C:/MEDSLIKE/outputsNEWall/vsiXYZ.npy', my_array)
print(my_array)"""


"""for i in tqdm(range(0,500)):
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
