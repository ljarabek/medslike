import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer, flatten
import numpy as np
import matplotlib.pyplot as plt
import CT as CT

BatchSize = 30
regularization = .01
conv1shape = [4,4,1,64]#conv2d filter [filter_height, filter_width, in_channels, out_channels]
bias1shape = [64]
seed = 42

input, answer = CT.getBatch(BatchSize,299)
#plt.imshow(input[0])
#plt.show()

'''print(len(input[0][0]))#279
print(len(input[0])) #194
print(len(input)) #10'''
input = np.array(input)
#input = input.reshape([BatchSize, len(input[0]), len(input[0,0]), 1])
input = np.expand_dims(input, 3)
print(input.shape)
#input = input.squeeze(0)


y = tf.placeholder(tf.float32, shape=[BatchSize,3])
x = tf.placeholder(tf.float32, shape =[BatchSize, len(input[0]), len(input[0,0]), 1])
with tf.name_scope('conv2d_1'):
     w1 = tf.get_variable(name = 'W1', shape = [8,8,1,16], initializer=xavier_initializer(uniform = True, seed=seed), regularizer=l2_regularizer(regularization))
     #w1 = tf.Variable(initial_value=)
     conv1 = tf.nn.conv2d(x,w1, strides = [1,1,1,1], padding = 'VALID') # strides so isto ko input
     b1 = tf.get_variable(name = 'b1', shape = [16], initializer=tf.constant_initializer(0.1)) #sizeout
     act1 = tf.nn.relu(conv1) #+ b1)
     #tf.summary.histogram("weights1", w1)
     #tf.summary.histogram("biases1", b1)
     #tf.summary.histogram('activations1', act1)
     #act1 = tf.nn.max_pool(act1, ksize=[1,4,4,1], strides=[1,4,4,1], padding = 'SAME')

with tf.name_scope('conv2d_2'):
    w2 = tf.get_variable(name='W2', shape= [4,4,16,32], initializer=xavier_initializer(uniform=True, seed=seed),
                         regularizer=l2_regularizer(regularization))
    # w1 = tf.Variable(initial_value=)
    conv2 = tf.nn.conv2d(act1, w2, strides=[1, 1, 1, 1], padding='VALID')  # strides so isto ko input
    b2 = tf.get_variable(name='b2', shape=[32], initializer=tf.constant_initializer(0.1))  # sizeout
    act2 = tf.nn.relu(conv2 + b2)
    #tf.summary.histogram("weights1", w1)
    #tf.summary.histogram("biases1", b1)
    #tf.summary.histogram('activations1', act1)
    #act2 = tf.nn.max_pool(act2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

with tf.name_scope('conv2d_3'):
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


with tf.name_scope('FC_1'):
    #act = fully_connected(act1,50, activation_fn=tf.nn.relu)
    #wfc = tf.get_variable(name= 'w_FC_1', shape = [97,140,64], initializer=xavier_initializer(uniform=True, seed = seed), regularizer=l2_regularizer(regularization)) #size in 64 --> size out 5
    #act = tf.scan(lambda x: tf.matmul(x,wfc, transpose_a=True), act1)
    flat = flatten(act2)
    activationFC1 = tf.layers.dense(
    flat,
    20,
    activation=None,
    use_bias=True,
    kernel_initializer=xavier_initializer(uniform=True, seed=seed),
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
#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(cost)
#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(cost)

merged = tf.summary.merge_all()

    #bfc = tf.get_variable(name='bfc', shape= [5], initializer=tf.constant_initializer(0.1))
    #act = tf.nn.relu(tf.matmul(act1, wfc) + bfc)

        #tf.summary.histogram('FC-biases', bfc)

arr = []
with tf.Session() as sess:
    writer = tf.summary.FileWriter('D:/Tboard/', sess.graph)
    sess.run(tf.global_variables_initializer()) #le tuki se inicializirajo weighti
    #for i in range(20):
    for i in range(150):
        input, answer = CT.getBatch(BatchSize, 299)
        inputx = np.expand_dims(input, 3)
        learning = 0.01
        if i > 10:
            learning = 0.001
        activation , output, costX, _ = sess.run([activationFC2, act3,  cost, train_step], feed_dict={y:answer, x:inputx, LR:learning})
        #writer.add_summary(summary=summarry)
        print('{} Cost: {} Answer: {} Prediction: {} '.format(i, costX, answer[4], activation[4]))
        arr = np.append(arr, costX)
        np.save('C:/test.npy', arr)
    #writer.
#print(cost)
#writer.add_summary(summary=summary)
#print(cost)
#acti = np.array(activation)
#print(acti.shape)
#print(acti)
#bla = np.array(output)
#print(bla.shape)
#bla = bla[5]
#bla = np.transpose(bla)
#bla = bla[4]
#bla = np.squeeze(bla, axis = 2)
#plt.imshow(bla)
#plt.show()
#answer = np.array(answer)
#print(answer.shape)
#print(answer)





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



