import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from IPython.display import display

#Import images into the program
'''
First retrieve the number of images in the dataset. Store them in a list to create 
the index used as labels. For example, the image as index 0 is goign to be the label 0

Then, iterate again over the images in order to assign each image to its label.
'''
persons=[]
labels=[]
testData=[]
imagesData=[]
#Iterate through the images to retrieve the possible labels and 
#open the images as matrix into the program
'''
for img in os.listdir(r'images/train/cat'):
    
    drtry = 'images/train/cat/'+img+'/'
    for img in os.listdir(drtry):
        myImg=Image.open(drtry+img)
        imagesData.append(np.asarray(myImg))
        if(img[5:7] in persons):
            continue
        else:
            persons.append(img[5:7])

print(persons)

for img in os.listdir(r'images/test'):
    drtry = 'images/test/'+img+'/'
    for img in os.listdir(drtry):
        myImg=Image.open(drtry+img)
        imagesData.append(np.asarray(myImg))
        if(img[5:7] in testData):
            continue
        else:
            testData.append(img[5:7])

print(*testData)
'''
'''for img in os.listdir('images/train/cat'):
    labels.append(persons.index(img[5:7]))


print(len(labels))
print(len(imagesData))
'''
print(len(persons))
#a = input('').split(" ")[0]
#print(a)

#train_data_dir = 'images/train'
#validation_data_dir = 'images/validation'

classifier = Sequential()

classifier.add(Convolution2D(32,3,2,input_shape=(68,77,3),activation = 'relu'))
classifier.add(Convolution2D(64,3,2,activation = 'relu'))
classifier.add(Convolution2D(64,3,2,activation = 'relu'))
classifier.add(Convolution2D(128,3,2,activation = 'relu'))

#classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim =128, activation = 'relu'))
beg=classifier.add(Dense(output_dim =30, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])




train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'images/train/cat',
    class_mode='categorical',
    target_size=(68,77),
    batch_size=32)

test_set = train_datagen.flow_from_directory(
    'images/test',
    target_size=(68,77),
    batch_size=32,
    class_mode='categorical')


classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=10,
    validation_data=test_set,
    validation_steps=800,
    callbacks=[TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='batch')])

#display()

classifier.save_weights("CNN/model_w.h5")
classifier.save_weights("CNN/model.h5")
plot_model(classifier, to_file='CNN/class.png')

