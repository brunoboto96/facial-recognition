import tensorflow.contrib.layers as nn
import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS

#
#normalize input
#batch norm
#leaky relu
#drpout
#regularization
#dataaugmentation

class CNN():
    def __init__(self, sess, model_v, test=False):
        self.sess=sess
        self.X=tf.placeholder(tf.float32, shape=[None, 77, 68, 1])
        self.num_outputs=30
        self.model_folder='CNN'
        self.model_v=model_v


        self.buildGraph()
        self.buildLoss()
        self.accuracy()

        self.saver=tf.train.Saver()
        if (test):
            self.saver.restore(self.sess, self.model_folder+"/"+self.model_v+"graph.ckpt")
        else:
            self.tensorboardStats()
            self.sess.run(tf.global_variables_initializer())



    def buildGraph(self):
        x_norm=self.X/255.
        self.l1=nn.conv2d(x_norm, 32, 7, stride=2)
        self.l2=nn.conv2d(self.l1, 64, 7, stride=2)
        self.l3=nn.conv2d(self.l2, 128, 5, stride=2)
        #self.l4=nn.conv2d(self.l3, 256, 3, stride=2)

        #flat the matrix
        self.flatten=nn.flatten(self.l3)

        self.l5=nn.fully_connected(self.flatten, 256)

        self.out=nn.fully_connected(self.l5, self.num_outputs, activation_fn=tf.nn.softmax)
    
    def buildLoss(self):
        self.labels=tf.placeholder(tf.int64, name='labels')
        #convert to 1 hot encode
        self.hot_encoded=tf.one_hot(self.labels, self.num_outputs)
        with tf.variable_scope('loss'):
            self.loss=tf.reduce_mean(-tf.reduce_sum(self.hot_encoded*tf.log(self.out + 1e-9) + (1-self.hot_encoded)*tf.log(1-self.out + 1e-9), axis=-1))
        
        with tf.variable_scope('optimizer'):
            self.opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    def accuracy(self):
        self.predicts=tf.argmax(self.out, axis=-1)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.predicts, self.labels), tf.float32))

    def tensorboardStats(self):
        self.file=tf.summary.FileWriter(self.model_folder+'/'+self.model_v, self.sess.graph)
        
        self.training=tf.summary.merge([
            tf.summary.scalar('loss', self.loss)
        ])

        self.testing=tf.summary.merge([
            tf.summary.scalar('accuracy', self.accuracy)
        ])

    def save(self):
        self.saver.save(self.sess, self.model_folder+'/'+self.model_v+'graph.ckpt')
