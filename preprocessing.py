import SimpleITK as sitk
import numpy as np
import CT
from tqdm import tqdm
import matplotlib.pyplot as plt
import string as st
import tensorflow as tf
import tensorflow.contrib.keras as tfk
import tensorflow.contrib.input_pipeline as tfi
import scipy.ndimage as spi
import os


def reject_outliers_and_standardize(data, m=2.0):
    data[abs(data - np.mean(data)) > m * np.std(data)] = 0
    m = np.mean(data)
    s = np.std(data)
    data[data==0] = m
    for idx, i in tqdm(np.ndenumerate(data)): data[idx] = (i-m)/(s+0.000001)
    data[abs(data - np.mean(data)) > m * np.std(data)] = 0
    return data







