import numpy as np
import CT
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tensorflow as tf
from sklearn import preprocessing
import scipy.spatial as sps
import scipy.ndimage as spi
from tqdm import tqdm
from pprint import pprint
import multiprocessing
from time import time
t1 = time()
print('mark')
arrN = []
for i in tqdm(range(50)):
    x = CT.SurfaceAsCoordinates(i)
    arrN.append(x)

np.save(arr = arrN, file = 'D:/MEDSLIKE/numpy/surface.17/vse.npy')
t2 = time()

print('noparalel {} , paralel: {}'.format(t2-t1, t11-t1))

#np.save('D:/MEDSLIKE/numpy/surface.17/vse.npy')
#np.save('D:/MEDSLIKE/numpy/povrsina_prvih_300.npy')
