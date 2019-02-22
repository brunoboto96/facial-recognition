'''
This script is used to use PCA on the dataset and bla bla bla
'''
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from preprocessing import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
X_train, X_test, y_train, y_test=preprocessing('images')

#flat each image
X_train=np.reshape(X_train, (X_train.shape[0], -1))
#Normalize images
X_train=(X_train.astype(float)-128)/128

def pca(comps, data):
    print(data.shape)
    pca = PCA(n_components=comps)
    pca.fit(data)
    #output comp-dimensional version of each image
    out = pca.transform(data)

    kmeans = KMeans(n_clusters=30, random_state=0).fit(out)

    acc=len(np.argwhere(kmeans.labels_==y_train))/data.shape[0]

    return acc

    '''
    colors = cm.rainbow(np.linspace(0, 1, 30))
    ax = plt.subplot(1, 1, 1)
    for idx, label in enumerate(y_train):
        x,y=out[idx]
        ax.plot(x,y, 'o', label=label)
    ax.legend(fontsize='small')
    plt.show()
    '''

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

components=[i for i in range(100, 5236, 100)]
print(components)
for c in components:
    acc=pca(c, X_train)
    print('components:', c, 'accuracy', acc)

