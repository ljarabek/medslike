import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer, flatten
import numpy as np
import layers as layers
import matplotlib.pyplot as plt
import pouya as CT
import configparser
from time import time
from pprint import pprint
from time import time
import scipy.io as sio
import os
import re as re





# THE MODEL - wide residual network*

config = configparser.ConfigParser()
config.read('config.ini')
cfg = config['DEFAULT']
BatchSize = int(cfg['batchsize'])
TrainMaxIndex = int(cfg['trainMaxIndex'])
testFrom = int(cfg['testFrom'])
testTo = int(cfg['testTo'])
path = cfg['MAT_path']

regularization = float(cfg['regularization'])

data = sio.loadmat(path)

X_train = data['X_train']
Y_train = data['Y_train']
X_pred = data['X_pred']
X_pred = (X_pred - np.mean(X_train)) / (np.std(X_train) + 0.000001)
#cfgCrop = config['CROP']
#yFrom = int(cfgCrop['yFrom'])
#yTo = int(cfgCrop['yTo'])


seed = 42

input, answer = CT.getBatch()   # placeholders
epoch = X_train.shape[0]
features = Y_train.shape[1]
input = np.array(input)
input = np.expand_dims(input, 3)

print('INPUT SHAPE: {}'.format(input.shape))


#MAKE directory for results:


SaveTo = 'c:/PouyaResults/' + "regularization_" + str(regularization) + "batch_" + str(BatchSize) + re.findall(r'(fx[0-9])',path)[0] + "_" + str(time()) + '/'
if not os.path.exists(SaveTo):
    os.makedirs(SaveTo)
    os.makedirs(SaveTo + "/y_pred/")


phase_train_bool = False
phase_train = tf.placeholder(tf.bool, name='phase_train')
y = tf.placeholder(tf.float32, shape=[BatchSize,features])  # coordinate input
x = tf.placeholder(tf.float32, shape =[BatchSize, len(input[0]), len(input[0,0]), 1])  # surface input
with tf.name_scope('conv2d_1'):
     w1 = tf.get_variable(name = 'W1', shape = [3,3,1,16], initializer=xavier_initializer(uniform = True, seed=seed), regularizer=l2_regularizer(regularization))
     conv1 = tf.nn.conv2d(x,w1, strides = [1,1,1,1], padding = 'VALID')
     conv1  = layers.batch_norm(conv1,16, phase_train)
     act1 = tf.nn.relu(conv1)

with tf.name_scope('BLOCK1'):
    w2 = tf.get_variable(name='W2', shape= [3,3,16,32], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))

    w21 = tf.get_variable(name='W2.1', shape= [1,1,16,32], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv2 = tf.nn.conv2d(act1, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = layers.batch_norm(conv2, 32, phase_train)
    conv2 = tf.nn.relu(conv2)
    w22 = tf.get_variable(name='W21', shape=[3, 3, 32, 32], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv2 = tf.nn.conv2d(conv2, w22, strides=[1, 1, 1, 1], padding='SAME')
    conv21 = tf.nn.conv2d(act1, w21, strides=[1, 1, 1, 1], padding='SAME') #!! 19.4.2018: strides 1111 -> 1221 ???? *results were obtained with 1,1,1,1
    act3 = conv2 + conv21

with tf.name_scope('BLOCK2'):
    conv3  = layers.batch_norm(act3, 32, phase_train)
    conv3 = tf.nn.relu(conv3)
    w3 = tf.get_variable(name='W3', shape=[3, 3, 32, 32], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv3 = tf.nn.conv2d(conv3, w3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = layers.batch_norm(conv3, 32, phase_train)
    conv3 = tf.nn.relu(conv3)
    w31 = tf.get_variable(name='W31', shape=[3, 3, 32, 32], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv3 = tf.nn.conv2d(conv3, w31, strides=[1, 1, 1, 1], padding='SAME')
    act4 = conv3 + act3


act4 = layers.batch_norm(act4, 32, phase_train)
act4 = tf.nn.relu(act4)

with tf.name_scope('BLOCK3'):
    w4 = tf.get_variable(name='W4', shape=[3, 3, 32, 64], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv4 = tf.nn.conv2d(act4, w4, strides=[1, 2, 2, 1], padding='SAME')
    conv4 = layers.batch_norm(conv4, 64, phase_train)
    conv4 = tf.nn.relu(conv4)
    w41 = tf.get_variable(name='W41', shape=[3, 3, 64, 64], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv4 = tf.nn.conv2d(conv4, w41, strides=[1, 1, 1, 1], padding='SAME')
    w42 = tf.get_variable(name='W4.2', shape=[1, 1, 32, 64], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv41 = tf.nn.conv2d(act4, w42, strides=[1, 2, 2, 1], padding='SAME')
    act5 = conv41 + conv4

with tf.name_scope('BLOCK4'):
    conv5 = layers.batch_norm(act5, 64, phase_train)
    conv5 = tf.nn.relu(conv5)
    w5 = tf.get_variable(name='W5', shape=[3, 3, 64, 64], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv5 = tf.nn.conv2d(conv5, w5, strides=[1, 1, 1, 1], padding='SAME')
    conv5 = layers.batch_norm(conv5, 64, phase_train)
    conv5 = tf.nn.relu(conv5)
    w51 = tf.get_variable(name='W51', shape=[3, 3, 64, 64], initializer=xavier_initializer(uniform=True, seed=seed),
                          regularizer=l2_regularizer(regularization))
    conv5 = tf.nn.conv2d(conv5, w51, strides=[1, 1, 1, 1], padding='SAME')
    act6 = conv5 + act5

act6 = layers.batch_norm(act6, 64, phase_train)
act6 = tf.nn.relu(act6)
with tf.name_scope('BLOCK5'):
    w6 = tf.get_variable(name='W6', shape=[3, 3, 64, 128], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv6 = tf.nn.conv2d(act6, w6, strides=[1, 2, 2, 1], padding='SAME')
    conv6 = layers.batch_norm(conv6, 128, phase_train)
    conv6 = tf.nn.relu(conv6)
    w61 = tf.get_variable(name='W61', shape=[3, 3, 128, 128], initializer=xavier_initializer(uniform=True, seed=seed),
                          regularizer=l2_regularizer(regularization))
    conv6 = tf.nn.conv2d(conv6, w61, strides=[1, 1, 1, 1], padding='SAME')
    w62 = tf.get_variable(name='W6.2', shape=[1, 1, 64, 128], initializer=xavier_initializer(uniform=True, seed=seed),
                          regularizer=l2_regularizer(regularization))
    conv61 = tf.nn.conv2d(act6, w62, strides=[1, 2, 2, 1], padding='SAME')
    act7 = conv61 + conv6
with tf.name_scope('BLOCK6'):

    conv7 = layers.batch_norm(act7, 128, phase_train)
    conv7 = tf.nn.relu(conv7)
    w7 = tf.get_variable(name='W7', shape=[3, 3, 128, 128], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv7 = tf.nn.conv2d(conv7, w7, strides=[1, 1, 1, 1], padding='SAME')
    conv7 = layers.batch_norm(conv7, 128, phase_train)
    conv7 = tf.nn.relu(conv7)
    w71 = tf.get_variable(name='W71', shape=[3, 3, 128, 128], initializer=xavier_initializer(uniform=True, seed=seed),
                          regularizer=l2_regularizer(regularization))
    conv7 = tf.nn.conv2d(conv7, w71, strides=[1, 1, 1, 1], padding='SAME')
    act8 = conv7 + act7
with tf.name_scope('BLOCK_X'):
    act8 = layers.batch_norm(act8, 128, phase_train)
    act8  = tf.nn.relu(act8)
    #act8 = tf.nn.avg_pool(act8, ksize=[1, 8, 8, 1], strides =[1,1,1,1], padding = 'VALID') #Keep it valid - keep it real


with tf.name_scope('FC_1'):
    #act = fully_connected(act1,50, activation_fn=tf.nn.relu)
    #wfc = tf.get_variable(name= 'w_FC_1', shape = [97,140,64], initializer=xavier_initializer(uniform=True, seed = seed), regularizer=l2_regularizer(regularization)) #size in 64 --> size out 5
    #act = tf.scan(lambda x: tf.matmul(x,wfc, transpose_a=True), act1)
    flat = flatten(act8)
    activationFC1 = tf.layers.dense(
    flat,
    128, #20!
    activation=None,
    use_bias=False,
    kernel_initializer=xavier_initializer(uniform=True, seed=seed),
    #kernel_initializer=tf.initializers.truncated_normal(mean = 0, stddev= 0.001),
    bias_initializer=tf.constant_initializer(0.1),
    kernel_regularizer=l2_regularizer(regularization),
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)

with tf.name_scope('FC_2'):
    #act = fully_connected(act1,50, activation_fn=tf.nn.relu)
    #wfc = tf.get_variable(name= 'w_FC_1', shape = [97,140,64], initializer=xavier_initializer(uniform=False, seed = seed), regularizer=l2_regularizer(regularization)) #size in 64 --> size out 5
    #act = tf.scan(lambda x: tf.matmul(x,wfc, transpose_a=True), act1)
    #flat = flatten(act3)
    activationFC2 = tf.layers.dense(
    #tf.contrib.layers.maxout(activationFC1, num_units=20)
    activationFC1,
    features,
    activation=None,
    use_bias=False, #True
    kernel_initializer=xavier_initializer(uniform=True, seed=seed),
    bias_initializer=tf.constant_initializer(0.1),
    kernel_regularizer=l2_regularizer(regularization),
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)


cost = tf.reduce_mean(tf.nn.l2_loss(activationFC2-y))

LR = tf.placeholder(tf.float32, [])




train_step = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)


merged = tf.summary.merge_all()



        #tf.summary.histogram('FC-biases', bfc)

arr = []
testarr = []

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #*ONLY HERE ARE THE WEIGHTS INITIALIZED
    a=0
    i_best = 1e5
    sTe, sTr = 1e8, 1e8  # placeholders
    costs = []  # placeholder
    learning = 0.01  #0001 TO DO: try with 3e-4, best learning rate
    for i in range(20000): #20k
        a+=1
        input, answer = CT.getBatch()
        inputx = np.expand_dims(input, 3)
        activation , output, costX, _ = sess.run([activationFC2, act8,  cost, train_step], feed_dict={y:answer, x:inputx, LR:learning, phase_train:True})
        tinput, tanswer = CT.getValBatch()
        inferLocation, costTest = sess.run([activationFC2, cost], feed_dict={y:tanswer, x:np.expand_dims(tinput, 3), LR:learning, phase_train:False})

        np.save(SaveTo + 'Costs_batch_train_test.npy', arr=costs)

        if i%700 == 0:
            learning *= 0.7

        if costTest<sTe and costX < sTr and a>500:
            sTe = costTest
            sTr = costX
            i_best=i
            print('APPENDED&SAVED')
            saver.save(sess, SaveTo + 'BestModel.ckpt')
            np.save(SaveTo + 'TrainBatch.npy', (activation, answer, costX))
            np.save(SaveTo + 'TestSetResults.npy', (inferLocation, tanswer, costTest))
            Y_pred = []
            for d in range(X_train.shape[0] + 1 - BatchSize):
                #stop  =  (i + 3) %140
                inferLocation = sess.run([activationFC2],
                                                   feed_dict={y: tanswer, x: np.expand_dims(X_pred[d:d+30], 3), LR: learning,
                                                              phase_train: False})
                Y_pred.append(inferLocation)
            np.save(SaveTo + 'Y_pred.npy', Y_pred)
            np.save(SaveTo + "/y_pred/Y_pred{}.npy".format(i), Y_pred)


        if(i%50==0):
            print('{} TRAINCost: {} TESTCost: {} LR: {}'.format(i, costX, costTest, learning))
            costs.append([costX, costTest])
        if(i_best+3000<i):
            print("overfitting detected, breaking the loop")
            break
CT.saveResults(SaveTo + 'Y_pred.npy', SaveTo + 'Y_pred_processed.npy')


    #writer.
"""bla = np.array(output) ???
print(output.shape)
print(bla.shape)
bla = bla[5]
bla = np.transpose(bla)
bla = bla[4]
#bla = np.squeeze(bla, axis = 2)
plt.imshow(bla)
plt.show()"""


