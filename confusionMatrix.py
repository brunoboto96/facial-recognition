import tensorflow as tf
from preprocessing import preprocessing
from agent import CNN
import numpy as np

import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
plotly.tools.set_credentials_file(username='steveoo', api_key='sYytnlmD1nhBjYwhADsO')
import plotly.graph_objs as go

X_train, X_test, y_train, y_test=preprocessing('images')
#Create the session where the graph can be run
with tf.Session() as sess:
    #create the graph
    cnn=CNN(sess, '', test=True)
    confusionMatrix=np.zeros((30,30)).astype(int)
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
        confusionMatrix[pred['label']][pred['predicted']] +=1
        print("true: {}, predicted: {}".format(pred['label'], pred['predicted']))
        print(pred['img'][:,:,0].shape)
        #plt.imshow(pred['img'][:,:,0], cmap='gray')
        #plt.show()

    gap=1/(np.max(confusionMatrix)+1)
    colorscale=[]
    colors=['rgb(0, 0, 110)', 'rgb(0, 147, 255)','rgb(191, 99, 9)', 'rgb(235, 213, 6)','rgb(254, 0, 0)']

    start=0
    for i in range(5):
        end=start+gap
        colorscale.append([start, colors[i]])
        colorscale.append([end, colors[i]])
        start=end

    x=[i+1 for i in range(30)]
    y=x[::-1]
    trace = go.Heatmap(z=confusionMatrix,
                       x=x,
                       y=y,
                       xgap=3,
                       ygap=3,
                       colorscale=colorscale)
    data=[trace]
    layout = go.Layout(
    xaxis=dict(
        title='Prediction',
        tickmode='linear',
        side='top'
    ),
    yaxis=dict(
        title='Label',
        autorange='reversed',
        tickmode='linear'
    )
    )
    fig=go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='basic-heatmap')

    print(confusionMatrix)