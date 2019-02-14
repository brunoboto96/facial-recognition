import tensorflow.contrib.layers as nn
import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS
'''
(10, 77, 68, 1)
(10, 77, 68, 32)
(10, 39, 34, 32)
(10, 20, 17, 64)
(10, 10, 9, 64)
(10, 4, 4, 64)
(10, 1024)
(10, 256)
(10, 30)
'''

class CNN():
    def __init__(self, sess):
        self.sess=sess
        self.X=tf.placeholder(tf.float32, shape=[None, 77, 68, 1])
        self.num_outputs=30
        self.model_folder='CNN'

        self.buildGraph()
        self.buildLoss()
        self.tensorboardStats()
        self.sess.run(tf.global_variables_initializer())

    def buildGraph(self):
        x_norm=self.X/255.
        self.l1=nn.conv2d(x_norm, 32, 5)
        self.l2=nn.conv2d(self.l1, 32, 5, stride=2)
        self.l3=nn.conv2d(self.l2, 64, 5, stride=2)
        self.l4=nn.conv2d(self.l3, 64, 3, stride=2)
        self.l4_pool=nn.max_pool2d(self.l4, 3)

        #flat the matrix
        self.flatten_l4=nn.flatten(self.l4_pool)

        #apply dropout to avoid overfit
        drop=nn.dropout(self.flatten_l4)

        self.l5=nn.fully_connected(drop, 256)

        self.out=nn.fully_connected(self.l5, self.num_outputs, activation_fn=tf.nn.softmax)
    
    def buildLoss(self):
        self.labels=tf.placeholder(tf.int32)
        #convert to 1 hot encode
        self.hot_encoded=tf.one_hot(self.labels, self.num_outputs)
        self.loss=tf.reduce_mean(-tf.reduce_sum(self.hot_encoded*tf.log(self.out + 1e-9) + (1-self.hot_encoded)*tf.log(1-self.out + 1e-9), axis=-1))
        #self.loss=self.hot_encoded*tf.log(self.out + 1e-9) + (1-self.hot_encoded)*tf.log(1-self.out + 1e-9)

        self.opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
    
    def tensorboardStats(self):
        self.file=tf.summary.FileWriter(self.model_folder, self.sess.graph)
        
        self.training=tf.summary.merge([
            tf.summary.scalar('loss', self.loss)
        ])

        self.avgRew=tf.placeholder(tf.float32)
        self.testing=tf.summary.merge([
            tf.summary.scalar('accuracy', self.avgRew)
        ])

