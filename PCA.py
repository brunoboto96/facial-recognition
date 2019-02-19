'''
This script is used to use PCA on the dataset and bla bla bla
'''
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocessing import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
X_train, X_test, y_train, y_test=preprocessing('images')

#flat each image
X_train=np.reshape(X_train, (X_train.shape[0], -1))
#Normalize images
X_train=(X_train.astype(float)-128)/128

def PCA():
    pca = PCA(n_components=2)
    pca.fit(X_train)
    #output 2 dimensional version of each image
    out = pca.transform(X_train)

    colors = cm.rainbow(np.linspace(0, 1, 30))
    ax = plt.subplot(1, 1, 1)
    for idx, label in enumerate(y_train):
        x,y=out[idx]
        ax.plot(x,y, 'o', label=label)
    ax.legend(fontsize='small')
    plt.show()

def TSNE():
    tsne = TSNE(n_components=2)
    #output 2 dimensional version of each image
    out = tsne.fit_transform(X_train)

    colors = cm.rainbow(np.linspace(0, 1, 30))
    ax = plt.subplot(1, 1, 1)
    for idx, label in enumerate(y_train):
        x,y=out[idx]
        ax.plot(x,y, 'o', label=label)
    ax.legend(fontsize='small')
    plt.show()