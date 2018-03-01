import SimpleITK as sitk
import numpy as np
import configparser
from tqdm import tqdm
import matplotlib.pyplot as plt
import string as st
import tensorflow as tf
import tensorflow.contrib.keras as tfk
import tensorflow.contrib.input_pipeline as tfi
import scipy.ndimage as spi
import os
from preprocessing import reject_outliers_and_standardize

vse=np.load('C:/MEDSLIKE/numpy/surface.17/vse.npy') #slike
#povp = np.mean(vse,0)
#stdd = np.std(vse,0)
#vse = reject_outliers_and_standardize(vse, 2.5)



config = configparser.ConfigParser()
config.read('config.ini')
cfg = config['DEFAULT']
BatchSize = int(cfg['batchSize'])
regularization = float(cfg['regularization'])

cfgCrop = config['CROP']
yFrom = int(cfgCrop['yFrom'])
yTo = int(cfgCrop['yTo'])
xFrom = int(cfgCrop['xFrom'])
xTo = int(cfgCrop['xTo'])

kvse = np.load('C:/MEDSLIKE/numpy/xyzTUMORJAzaPRVIH300slik.npy') #koordinate
kpovp = np.mean(kvse,0)
kstdd = np.std(kvse, 0) #!! išči ~absolutne odmike (y je nepomembn!) BREZ ali Z 0



def GetDir (consecutive_number, indir = 'C:\MEDSLIKE'):
    for root, dirs, filenames in os.walk(indir):
        return os.path.join(root, filenames[consecutive_number])

def GetImage(consecutive_number, indir = 'C:\MEDSLIKE'):
    return sitk.ReadImage(GetDir(consecutive_number,indir))

def LoadCTAsArray(consecutive_number, indir = 'C:\MEDSLIKE'):
    path = GetDir(consecutive_number, indir)
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def LoadCTAsArrayView(consecutive_number, indir = 'C:\MEDSLIKE'):
    path = GetDir(consecutive_number, indir)
    return sitk.GetArrayViewFromImage(sitk.ReadImage(path))  #ArrayView ne moreš editat, je pa bolj efficient


def SliceAnimation(slicexx, frames, readDir = 'C:/MEDSLIKE/', destinationImage = sitk.ReadImage('C:/MEDSLIKE/main2.mha')):   #dopolni sliko destinationImage, v kateri je vsaka rezina slicexx iz zaporednih slik; število rezin = frames
    for i in tqdm(range(frames)):
        Barr = LoadCTAsArray(i, readDir)
        for x in range(len(Barr[slicexx])):  # zbrali smo si slice slicexxx (116 prej, zdej 104)
            for y in range(len(Barr[slicexx, x])):
                destinationImage[y, x, i] = np.float32(Barr[slicexx, x, y]).item()

def AddLabelsToSlices():
    bla = sitk.ReadImage('C:/MEDSLIKE/main2.mha')
    arr = sitk.GetArrayFromImage(bla)  # prebere sliko
    # Tarr = tf.convert_to_tensor(arr) #spremeni v tensor
    b=0
    for i in arr:
        phaseno = (b+15)%25   #to je kr ena funkcija, ki sm si zbral, za razdelitev dihanja v 25 faz*
        b+=1

        return i , phaseno       #vrne par - slika,faza dihanja

def CropImage(x1,x2,y1,y2,z1,z2, image):
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    z = abs(z1 - z2)
    newimg = sitk.Image(x, y, z, sitk.sitkFloat64)
    for i in range(x): #tqdm(range(x)):
        for j in range(y):
            for k in range(z):
                newimg[i, j, k] = image.GetPixel(i + x1, j + y1, k + z1)
                # print(hah.GetPixel(i,j,k))
    return newimg

def ShowArrayAsCT(arr, Name = None):
    im = sitk.GetImageFromArray(arr)
    sitk.Show(im, title = Name)

def ConvertArrayToBinary(arr, threshold, values = (-0.7,1)): #array must be 3D; če ne dela spremen tuple za default values v array --> () v []
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            for k in range(len(arr[0,0])):
                if (arr[i,j,k]>threshold):
                    arr[i,j,k] = values[1]
                else:
                    arr[i,j,k] = values[0]
    return arr

def ConvertImageToBinary(image, threshold):
    y = image.GetHeight()
    x = image.GetWidth()
    z = image.GetDepth()
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if(image.GetPixel(i,j,k)>threshold):
                    image[i,j,k] = 1.0
                else:
                    image[i,j,k] = 0.0
    return image

def GetMaxWeightedIndex(c, directory = 'C:/MEDSLIKE/outputs/{}.npy'): #vrne tuple xyz z koordinatami težišča slike
    haha = np.load(directory.format(c))  #prebere slike z verjetnosti/'logits' tumorja na lokacijah (output ročno-nastavljenega CNN-ja)
    #haha = haha.squeeze(0)                                  #zarad tensorflowa je prejšni array oblike 1,x,y,z,1 --> squeeze da dobimo xyz
    #haha = haha.squeeze(3)
    return spi.center_of_mass(haha)


def SurfaceAsCoordinates(consecutive_number, dirr='C:/MEDSLIKE', gradientThreshold = 0.17):  #Fukne ven array: z-->y , x-->x, y-->f(x,y)=VREDNOSTI
    img = GetImage(consecutive_number,indir=dirr)
    imgG = sitk.GradientMagnitude(img)
    arrG = sitk.GetArrayFromImage(imgG)
    arrNew = np.array(np.zeros((len(arrG), len(arrG[0,0])))) #prva koordinata bo X, druga pa Z!!

    for z in range(len(arrG)):
        for x in range(len(arrG[0,0])):
            for y in range(len(arrG[0])):
                if arrG[z,y,x]>gradientThreshold:
                    arrNew[z,x] = y
                    break
    #plt.imshow(arrNew)     ZA IZRISAT!
    #plt.show()
    np.save(arr = arrNew, file ='C:/MEDSLIKE/numpy/surface.17/{}.npy'.format(consecutive_number))
    return arrNew

#OLD standardization!!:
def standardize2D(arr):
    return np.nan_to_num((arr-povp)/(stdd+0.00001))
def standardizecoords(arr):
    return np.nan_to_num((arr-kpovp)/(kstdd+0.00001))


def getBatch(size = 10, maxsize = 299, minsize = 0):
    rn = np.random.randint(0, maxsize+1-minsize, size = (size))
    arr = []
    coordinates = []
    app = vse
    coo = np.load('C:/MEDSLIKE/numpy/xyzTUMORJAzaPRVIH300slik.npy')
    for i in rn:
        arr.append(app[i])
        coordinates.append(standardizecoords(coo[i]))
    return np.array(arr)[:,yFrom:yTo,xFrom:xTo], coordinates

def getBatchTest(min = 299, max = 335):
    arr = []
    coordinates = []
    for i in range(min,max):
        arr.append(vse[i])
        coordinates.append(standardizecoords(np.load('C:/MEDSLIKE/XYZ/train500.npy')[i]))
    return np.array(arr)[:,yFrom:yTo,xFrom:xTo], coordinates

def getBatchOLD(size = 10, maxsize = 299, minsize = 0):   #return arr, coordinates
    rn = np.random.randint(0, maxsize+1-minsize, size = (size))
    arr = []
    coordinates = []
    app = np.load('C:/MEDSLIKE/numpy/surface.17/vse.npy')
    coo = np.load('C:/MEDSLIKE/numpy/xyzTUMORJAzaPRVIH300slik.npy')
    for i in rn:
        arr.append(standardize2D(app[i]))
        coordinates.append(standardizecoords(coo[i]))

    return arr, coordinates

def getBatchTestOLD(size = 10, min = 299, max = 335):
    arr = []
    coordinates = []
    for i in np.random.randint(min,max, size = size):
        arr.append(standardize2D(np.load('C:/MEDSLIKE/numpy/surface.17/{}.npy'.format(i))))
        coordinates.append(standardizecoords(np.load('C:/MEDSLIKE/XYZ/train500.npy')[i]))
    return arr, coordinates

#print(kstdd)
#print(kpovp)
#bla = sitk.ReadImage('C:/MEDSLIKE/main2.mha')
#sitk.Show(bla)

"""def NaredEnArray():
    A = LoadCTAsArray(0)[55]
    print()
    for i in range(10):
        x = i+1
        np.append(A ,LoadCTAsArray(x)[55],axis = 1)        TOLE NEVEM, KA JE
    #sitk.Cast(sitk.ReadImage('C:/00000.mha'),A)
    return A"""

"""
def crop2DArr(arrayToCrop, ymin, ymax, xmin, xmax): ##REDUNDANTNO (uporabi raj np.array(haha)[:,:,from:to,:,:]
    x = abs(ymin - ymax)
    y = abs(xmin - xmax)
    new_array = np.zeros(shape = (x,y))
    for i in range(x):
        for j in range(y):
            bla = arrayToCrop[ymin+i, j + xmin]
            new_array[i,j] =bla
    return new_array
"""

#B = NaredEnArray()
"""_max = np.max(B)
for i in range(len(B)):
    for j in range(len(B[i])):
        for k in range(len(B[i,j])):
            B[i,j,k] /= _max

B *= 255
np.save(B, 'C:/3dimage.npy')"""

"""np.save("C:/nparray", LoadCTAsArray(5)[55]) < ups  to je narobe (zamenji npyrray pa loadCTS...
print(LoadCTAsArray(5)[55])"""
#arr = LoadCTAsArray(5)
#print(arr.dtype)
#tfk.preprocessing