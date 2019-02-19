import numpy as np
from agent import CNN
import tensorflow as tf
from preprocessing import preprocessing, dataAugmentation

X_train, X_test, y_train, y_test=preprocessing('images')


#Create the session where the graph can be run
with tf.Session() as sess:
    #create the graph
    cnn=CNN(sess, 'v1')

    step=0
    #train the model 1000 times
    for i in range(1000+1):
        print('Training epoch ({}/{})'.format(i, 1000+1))
        #Sample 32 examples (mini-batch) from the training images to train the network
        idxs=np.random.choice(X_train.shape[0], 32)
        batchInput=X_train[idxs]
        batchLabels=y_train[idxs]

        #Augment the real images like crop, flip, noise, translate, rotate..
        #batchTrain=dataAugmentation(X_train[startBatch:endBatch])

        #feed the 32 examples as well their labels into the model and train it
        _, summ=sess.run([cnn.opt, cnn.training], feed_dict={cnn.X: batchInput,
                                                            cnn.labels: batchLabels})
        cnn.file.add_summary(summ, step)
        step+=1
        #every 5 times, use the test examples to see if the model is learning
        if i%5==0:
            print('testing..')

            idxs=np.random.choice(X_test.shape[0], 32)
            batchInput=X_test[idxs]
            batchLabels=y_test[idxs]

            summ=sess.run(cnn.testing, feed_dict={cnn.X: batchInput,
                                                  cnn.labels: batchLabels})
            cnn.file.add_summary(summ, step)

            #Save the model weights
            cnn.save()