import numpy as np
import tensorflow as tf 
from agent import CNN
'''
This script is used to the the CNN if othere aren't bug
'''

inputImg=np.random.randint(0,255,(32,77,68,1))
labels=np.random.randint(0,30,(32)).astype(int)
with tf.Session() as sess:
    print(inputImg.shape)
    cnn=CNN(sess)
    for i in range(500):
        _, summ=sess.run([cnn.opt, cnn.training], feed_dict={cnn.X: inputImg,
                                                             cnn.labels: labels})
        cnn.file.add_summary(summ, i)

