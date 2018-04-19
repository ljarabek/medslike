import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer, flatten
import numpy as np
import layers as layers
import matplotlib.pyplot as plt
import CT as CT
import configparser
from pprint import pprint
from time import time

#THIS IS THE MODEL

config = configparser.ConfigParser()
config.read('config.ini')
cfg = config['DEFAULT']
BatchSize = int(cfg['batchSize'])
TrainMaxIndex = int(cfg['trainMaxIndex'])
testFrom = int(cfg['testFrom'])
testTo = int(cfg['testTo'])

regularization = float(cfg['regularization'])


cfgCrop = config['CROP']
yFrom = int(cfgCrop['yFrom'])
yTo = int(cfgCrop['yTo'])


seed = 42

input, answer = CT.getBatch(BatchSize,250)


input = np.array(input)
input = np.expand_dims(input, 3)

print('INPUT SHAPE: {}'.format(input.shape))

phase_train_bool = False
phase_train = tf.placeholder(tf.bool, name='phase_train')
y = tf.placeholder(tf.float32, shape=[BatchSize,3])
x = tf.placeholder(tf.float32, shape =[BatchSize, len(input[0]), len(input[0,0]), 1])
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
    conv21 = tf.nn.conv2d(act1, w21, strides=[1, 1, 1, 1], padding='SAME') #!!!!!!!!! 19.4.2018: strides 1111 -> 1221 ????
    act3 = conv2 + conv21

with tf.name_scope('BLOCK2'):
    #conv3 = tf.nn.selu(act3)
    conv3  = layers.batch_norm(act3, 32, phase_train)
    conv3 = tf.nn.relu(conv3)
    w3 = tf.get_variable(name='W3', shape=[3, 3, 32, 32], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv3 = tf.nn.conv2d(conv3, w3, strides=[1, 1, 1, 1], padding='SAME')
    #conv3 = tf.nn.selu(conv3)
    conv3 = layers.batch_norm(conv3, 32, phase_train)
    conv3 = tf.nn.relu(conv3)
    w31 = tf.get_variable(name='W31', shape=[3, 3, 32, 32], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv3 = tf.nn.conv2d(conv3, w31, strides=[1, 1, 1, 1], padding='SAME')
    act4 = conv3 + act3

#act4 = tf.nn.selu(act4)
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
#act6 = tf.nn.selu(act6)
act6 = layers.batch_norm(act6, 64, phase_train)
act6 = tf.nn.relu(act6)
with tf.name_scope('BLOCK5'):
    w6 = tf.get_variable(name='W6', shape=[3, 3, 64, 128], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    conv6 = tf.nn.conv2d(act6, w6, strides=[1, 2, 2, 1], padding='SAME')
    #conv6 = tf.nn.selu(conv6)
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
    #conv7 = tf.nn.selu(act7)
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
with tf.name_scope('FOTOFINISH'):
    act8 = layers.batch_norm(act8, 128, phase_train)
    act8  = tf.nn.relu(act8)
    act8 = tf.nn.avg_pool(act8, ksize=[1, 8, 8, 1], strides =[1,1,1,1], padding = 'VALID') #- TO JE BLO PREJ
    #act8 = flatten(act8)




"""with tf.name_scope('conv2d_3'):
    w3 = tf.get_variable(name='W3', shape= [4,4,32,64], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    # w1 = tf.Variable(initial_value=)
    conv3 = tf.nn.conv2d(act2, w3, strides=[1, 1, 1, 1], padding='VALID')  # strides so isto ko input
    b3 = tf.get_variable(name='b3', shape=[64], initializer=tf.constant_initializer(0.1))  # sizeout
    act3 = tf.nn.relu(conv3) #+ b3)
    #tf.summary.histogram("weights1", w1)
    #tf.summary.histogram("biases1", b1)
    #tf.summary.histogram('activations1', act1)
    act3 = tf.nn.max_pool(act3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
"""

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
    3,
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

"""with tf.name_scope('FC_3'):
    #act = fully_connected(act1,50, activation_fn=tf.nn.relu)
    #wfc = tf.get_variable(name= 'w_FC_1', shape = [97,140,64], initializer=xavier_initializer(uniform=True, seed = seed), regularizer=l2_regularizer(regularization)) #size in 64 --> size out 5
    #act = tf.scan(lambda x: tf.matmul(x,wfc, transpose_a=True), act1)
    activationFC3 = tf.layers.dense(
    activationFC2,
    3,
    activation=None,
    use_bias=True,
    kernel_initializer=xavier_initializer(uniform=True, seed=seed),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=l2_regularizer(regularization),
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)"""

cost = tf.reduce_mean(tf.nn.l2_loss(activationFC2-y))
#cost = tf.square(activationFC2-y)
#cost = tf.convert_to_tensor(cost)
LR = tf.placeholder(tf.float32, [])




train_step = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)
#train_step = tf.train.AdagradOptimizer(learning_rate=LR).minimize(cost)
#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(cost)

merged = tf.summary.merge_all()

    #bfc = tf.get_variable(name='bfc', shape= [5], initializer=tf.constant_initializer(0.1))
    #act = tf.nn.relu(tf.matmul(act1, wfc) + bfc)

        #tf.summary.histogram('FC-biases', bfc)

arr = []
testarr = []

saver = tf.train.Saver()
#builder = tf.saved_model.builder.SavedModelBuilder("C:/MEDSLIKE/TFMODEL/model")

with tf.Session() as sess:
    #writer = tf.summary.FileWriter('D:/Tboard/', sess.graph)
    sess.run(tf.global_variables_initializer()) #le tuki se inicializirajo weighti
    #for i in range(20):
    a=0
    sTe, sTr = 1e8, 1e8
    costs = []
    learning = 0.0001 #ena nula majn*
    for i in range(80000): #8k
        a+=1

        input, answer = CT.getBatch(BatchSize, TrainMaxIndex)
        inputx = np.expand_dims(input, 3)
        #0.0001 -- ful dobr, če ni v CT ln 18 - ,0 na konc
        #if i > 10:
            #learning = 0.001
        activation , output, costX, _ = sess.run([activationFC2, act8,  cost, train_step], feed_dict={y:answer, x:inputx, LR:learning, phase_train:True})
        #writer.add_summary(summary=summarry)
        tinput, tanswer = CT.getBatchTest(max = testTo, min = testFrom)
        inferLocation, costTest = sess.run([activationFC2, cost], feed_dict={y:tanswer, x:np.expand_dims(tinput, 3), LR:learning, phase_train:False})

        costs.append([costX, costTest])
        np.save('C:/MEDSLIKE/RESULTS/every50/costs.npy', arr=costs)
        if costTest<sTe and costX < sTr and a>500:
            sTe = costTest
            sTr = costX

            print('APPENDED&SAVED')
            saver.save(sess, 'C:/MEDSLIKE/RESULTS/every50/model.ckpt')

            np.save('C:/MEDSLIKE/RESULTS/every50/trainreg05.npy', (activation, answer, costX))
            np.save('C:/MEDSLIKE/RESULTS/every50/testreg05.npy', (inferLocation, tanswer, costTest))

        """if (a>20):
            test_c = testarr[:, 2]
            pprint(test_c)
            train_c = arr[:, 2]
            a=0
            old_mean = np.mean(train_c[i-a-19:i-a])
            new_mean = np.mean(train_c[i-a:])
            print('OLDMEAN {} \nNEWMEAN {}'.format(old_mean,new_mean))
            if(new_mean/old_mean < 1.2):
                learning = learning/10
                print('reduced LR to:{}'.format(learning))
            if(costX<min(train_c)):
                if(costTest<min(test_c)):
                    print('NEW RECORD! saving to C:/MEDSLIKE/TFMODEL/modelCheckpointBest.ckpt')
                    save_path = saver.save(sess, 'C:/MEDSLIKE/TFMODEL/modelCheckpointBest.ckpt')"""
        print('{} TRAINCost: {} TESTCost: {}'.format(i, costX, costTest))
    #writer.
bla = np.array(output)
print(output.shape)
print(bla.shape)
bla = bla[5]
bla = np.transpose(bla)
bla = bla[4]
#bla = np.squeeze(bla, axis = 2)
plt.imshow(bla)
plt.show()





def conv_layer(input, size_in, size_out, name = 'Conv'):
    with tf.name_scope(name):
        w1 = tf.get_variable(name='W1', shape=[5,5, size_in, size_out], initializer=xavier_initializer(uniform=False, seed=seed),
                             regularizer=l2_regularizer(regularization))
        # w1 = tf.Variable(initial_value=)
        conv1 = tf.nn.conv2d(input, w1, strides=[1, 1, 1, 1], padding='SAME')  # strides so isto ko input
        b1 = tf.get_variable(name='b1', shape=[size_out], initializer=tf.constant_initializer(0.1))  # sizeout
        act1 = tf.nn.relu(conv1 + b1)
        tf.summary.histogram("weights1", w1)
        tf.summary.histogram("biases1", b1)
        tf.summary.histogram('activations1', act1)
        return tf.nn.max_pool(act1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fully_connected(input, no_outputs, name = 'FC'):
    with tf.name_scope(name):
        act = tf.layers.dense(
            input,
            no_outputs,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=xavier_initializer(uniform=False, seed=seed),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=l2_regularizer(regularization),
            bias_regularizer=l2_regularizer(regularization),
            activity_regularizer=None,
            trainable=True,
            name=None,
            reuse=None
        )
        return act




