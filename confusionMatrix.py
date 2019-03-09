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
        #Retrieve the examples where the prediction doesn't match the label
        wrong_preds=np.argwhere(y_test[startBatch:endBatch]!=pred)
        if len(wrong_preds)!=0:
            #for each wrong prediction, retrieve the example img, the label and the prediction to use in the heatmap
            for idx in wrong_preds[0]:
                prediction={}
                prediction['label']=y_test[startBatch:endBatch][idx]
                prediction['predicted']=pred[0][idx]
                prediction['img']=X_test[startBatch:endBatch][idx]
                predictions.append(prediction)

    #Compute confusion matrix and tell the examples where the prediction is wrong
    for pred in predictions:
        confusionMatrix[pred['label']][pred['predicted']] +=1
        print("True person idx: {}, Predicted person idx: {}".format(pred['label'], pred['predicted']))
    '''
    #ONLY FOR CREATING THE IMAGE HERE
    preds=[predictions[12], predictions[-1]]
    f, axarr = plt.subplots(2,2)
    for i in range(2):
        axarr[i,0].set_xticks([])
        axarr[i,0].set_yticks([])
        axarr[i,0].set_title('Input Person '+ str(preds[i]['label']))
        axarr[i,0].imshow(preds[i]['img'][:,:,0])
        #get image of predicted person
        axarr[i,1].set_xticks([])
        axarr[i,1].set_yticks([])
        axarr[i,1].set_title('Predicted Person ' + str(preds[i]['predicted']))
        idx=np.argwhere(y_test==preds[i]['predicted'])[0][0]
        axarr[i,1].imshow(X_test[idx][:,:,0])
    plt.show()
    '''

    #Create heatmap
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
    trace = go.Heatmap(z=np.flip(confusionMatrix, axis=0),
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