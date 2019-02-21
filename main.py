import numpy as np
from agent import CNN
import tensorflow as tf
from preprocessing import preprocessing, dataAugmentation

X_train, X_test, y_train, y_test=preprocessing('images')


#Create the session where the graph can be run
with tf.Session() as sess:
    #create the graph
    cnn=CNN(sess, '')

    step=0
    #train the model 1000 times
    for i in range(2000+1):
        print('Training epoch ({}/{})'.format(i, 2000+1))
        #Sample 32 examples (mini-batch) from the training images to train the network
        idxs=np.random.choice(X_train.shape[0], 32)
        batchInput=X_train[idxs]
        batchLabels=y_train[idxs]

        #Augment the real images like crop, flip, noise, translate, rotate..
        batchInput=dataAugmentation(batchInput)

        #feed the 32 examples as well their labels into the model and train it
        _, summ=sess.run([cnn.opt, cnn.training], feed_dict={cnn.X: batchInput,
                                                            cnn.labels: batchLabels})
        cnn.file.add_summary(summ, step)
        step+=1

    test_accs=[]
    for startBatch in range(0, X_test.shape[0], 64):
        endBatch=startBatch+64
        batchInput= X_test[startBatch:endBatch]
        batchLabels=y_test[startBatch:endBatch]

        acc=sess.run(cnn.accuracy, feed_dict={cnn.X: batchInput,
                                                cnn.labels: batchLabels})
        test_accs.append(acc)

    print(np.mean(test_accs))
    cnn.save()