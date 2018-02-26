import tensorflow as tf
import tensorflow.contrib.keras as tfk
import numpy as np
from pprint import pprint
from sklearn.preprocessing import normalize, scale
import CT
import matplotlib.pyplot as plt

arr = np.load('c:/test.npy')
answers = scale(np.load('D:/MEDSLIKE/numpy/xyzTUMORJAzaPRVIH300slik.npy'))
#maš array tuplejev --> druga koordinata je x/y/z 0/1/2 prva pa primer !!! TAKO JE!!
answers = answers[0]
#answers = CT.crop2DArr(answers,118,397,17,211)
inputs = np.load('D:/MEDSLIKE/numpy/surface.17/vse.npy')#scale(np.load('D:/MEDSLIKE/numpy/surface.17/vse.npy'))
#pprint(inputs)
#pprint(inputs.shape)
"""
!!!TODO-  TLE VSE DELA!! :D :D :D
moraš zrezat input !!!

zračuni cost !!! DONE?

zračuni gradient !!! DONE?

določi batche !!!

treniraj!!!

...

PROFIT!!

"""
#tf.variable
y = tf.placeholder(tf.float32, shape=[3])
x = tf.placeholder(tf.float32, shape=(1,len(inputs[0]), len(inputs[0,0]),1)) #batch, height(y), width(x), channels
w1 = tf.Variable(tf.random_normal([16,16,1,25], mean=0)) #(filter)height (filter)width inchannel outchannel
b1 = tf.Variable(tf.random_normal(shape=[25]))
x1 = tf.nn.conv2d(x,w1,[1,2,2,1], padding = 'VALID')
x1 = tf.nn.max_pool(x1,ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
x1 = tf.nn.bias_add(x1, b1)
x1 = tf.nn.relu(x1)

w2 = tf.Variable(tf.random_normal([8,8,25,125], mean=0))
b2 = tf.Variable(tf.random_normal(shape=[125]))
x2 = tf.nn.conv2d(x1,w2,[1,2,2,1], padding = 'VALID')
x2 = tf.nn.max_pool(x2,ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
x2 = tf.nn.bias_add(x2,b2)
x2 = tf.nn.relu(x2)
#tf.nn.
'''
b3 = tf.Variable(tf.random_normal(shape=[625]))
w3 = tf.Variable(tf.random_normal([11,3,125,625], mean=0))
x3 = tf.nn.conv2d(x2,w3,[1,1,1,1], padding = 'VALID')
#x3 = tf.nn.max_pool(x3,ksize=[1,2,2,1], strides = [1,4,4,1], padding = 'VALID')
x3 = tf.nn.bias_add(x3,b3)
x3 = tf.nn.relu(x3)

#Fully connected:

b4 = tf.Variable(tf.random_normal(shape=[625]))
w4 = tf.Variable(tf.random_normal([1,27,625,3]))
x4 = tf.nn.conv2d(x3,w4,strides=[1,1,1,1], padding = 'VALID')
'''

#x4 = tf.nn.relu(x4)



cost = tf.reduce_mean(tf.nn.l2_loss(x2-y))
train = tf.train.AdamOptimizer().minimize(cost)
#cost = tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = cost)
#x1 =






with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    In = scale(inputs[0])
    arrx, cost, train = sess.run([x1, cost, train], feed_dict={x:np.expand_dims(np.expand_dims(In,axis=2),axis=0), y:answers})
print(arrx.shape)
print(cost)
print(train)
#print(costs)
n = arrx[0,:,:, 1]
pprint("{},,{},,{}".format(arrx[0,:,:, 1],arrx[0,:,:, 0],arrx[0,:,:, 2]))
#pprint(n.shape)
#arrx = arrx.squeeze(3)
#arrx = arrx.squeeze(0)
#plt.imshow(n)
#plt.show()
plt.imshow(inputs[0])
plt.show()







#print('dim 1 \t {0}  \ndim 2  \t {1}\ndim 3  \t {2}\ndim 4  \t {3}'.format(len(output), len(output[0]), len(output[0,0]), len(output[0,0,0])))