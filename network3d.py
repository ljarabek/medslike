import numpy as np
import tensorflow as tf
import CT as CT
from sklearn import preprocessing
from tqdm import tqdm
import SimpleITK as sitk

"""DOBI SLIKO TUMORJA: """
hah = CT.GetImage(10)
x1 = 171 #mere tumorja
x2 = 188
y1 = 233
y2 = 248
z1 = 117
z2 = 126
tumor = {'x':x2-x1 , 'y': y2-y1 , 'z':z2-z1}
kernel = CT.CropImage(171,188,233,248,117,126,hah) #GET kernelIMAGE = tumor
#kernel = CT.ConvertImageToBinary(kernel, 0.5)
#sitk.Show(kernel) #prikaže tumor
kernel = sitk.GetArrayFromImage(kernel)
kernel = CT.ConvertArrayToBinary(kernel, threshold=0.5)
#kernel  = preprocessing.scale(kernel) dela sam za 2d arraye
kernel = kernel.reshape([tumor['z'], tumor['y'], tumor['x'], 1, 1])

"""DOBI SLIKO PREGLEDANEGA PODROČJA"""

x1 = 160
x2 = 201 #41 (+20)
y1 = 220
y2 = 261 #41 (+20)
z1 = 100
z2 = 131 #31 (+15)
dimenzije = {'x':x2-x1 , 'y': y2-y1 , 'z':z2-z1}
def GetArea(No, x1=160,x2=201,y1=220,y2=261,z1=100,z2=131):
    imag = CT.GetImage(No)
    ima = CT.CropImage(x1,x2,y1,y2,z1,z2, imag)
    area = sitk.GetArrayFromImage(ima)
    area = area.reshape([1, dimenzije['z'], dimenzije['y'], dimenzije['x'], 1])
    return area


#kernel = np.zeros(shape = (1,16,16,16,1)) #TO ZBRIŠ!
strides = 1

w = tf.placeholder(tf.float32) #shape=[tumor['z'], tumor['y'], tumor['x'], 1, 1])
x = tf.placeholder(tf.float32, shape = [1, dimenzije['z'], dimenzije['y'], dimenzije['x'], 1])
y = tf.nn.conv3d(x, w, (1, strides, strides, strides, 1), padding = 'VALID')  #PREJ VALID
y = tf.nn.relu(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = np.zeros(shape = (1,23,27,25,1))
    for i in tqdm(range(590)):
        convoluted = sess.run(y, feed_dict={w:kernel, x:GetArea(0)})
        output = convoluted
        output = output.squeeze(0)
        output = output.squeeze(3)
        np.save('C:/MEDSLIKE/outputsNEWall/{}.npy'.format(i), output)

#np.save('c:/balbla.npy', convoluted)
CT.ShowArrayAsCT(output, Name='output')
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
dimenzije = {'x':x2-x1 , 'y': y2-y1 , 'z':z2-z1}
area = sitk.GetArrayFromImage(area)
area = area.reshape([1, dimenzije['z'], dimenzije['y'], dimenzije['x'], 1])
"""

"""
print("output shape : {}".format(output.shape))
print('input shape: {}'.format(area.shape))
print('kernel shape: {}'.format(kernel.shape))
output shape : (23, 27, 25)
input shape: (1, 31, 41, 41, 1)
kernel shape: (9, 15, 17, 1, 1)
"""