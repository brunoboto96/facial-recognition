import tensorflow as tf
from preprocessing import preprocessing
from agent import CNN
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test=preprocessing('images')
#Create the session where the graph can be run
with tf.Session() as sess:
    #create the graph
    cnn=CNN(sess, 'v10', test=True)
    predictions=[]
    #Iterate through the test examples 32 at the time
    for startBatch in range(0, X_test.shape[0], 32):
        endBatch=startBatch+32
        pred=sess.run([cnn.predicts], feed_dict={cnn.X: X_test[startBatch:endBatch],
                                                 cnn.labels: y_test[startBatch:endBatch]})
        wrong_preds=np.argwhere(y_test[startBatch:endBatch]!=pred)
        if len(wrong_preds)!=0:
            for idx in wrong_preds[0]:
                prediction={}
                prediction['label']=y_test[startBatch:endBatch][idx]
                prediction['predicted']=pred[0][idx]
                prediction['img']=X_test[startBatch:endBatch][idx]
                predictions.append(prediction)
    
    for pred in predictions:
        print("true: {}, predicted: {}".format(pred['label'], pred['predicted']))
        print(pred['img'][:,:,0].shape)
        plt.imshow(pred['img'][:,:,0], cmap='gray')
        plt.show()