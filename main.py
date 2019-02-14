import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from agent import CNN
import tensorflow as tf
#Import images into the program
'''
First retrieve the number of images in the dataset. Store them in a list to create 
the index used as labels. For example, the image as index 0 is goign to be the label 0

Then, iterate again over the images in order to assign each image to its label.


'''
persons=[]
labels=[]
imagesData=[]
#Iterate through the images to retrieve the possible labels and 
#open the images as matrix into the program
for img in os.listdir('images'):
    myImg=Image.open('images/'+img)
    imagesData.append(np.asarray(myImg))
    if(img[5:7] in persons):
        continue
    else:
        persons.append(img[5:7])


for img in os.listdir('images'):
    labels.append(persons.index(img[5:7]))

print(len(persons))
print(len(labels))
print(len(imagesData))

#divide in train-accuracy dataset (data augmentation)

x1=imagesData[:int(len(imagesData)*0.75)]
y1=labels[:int(len(labels)*0.75)]
x2=imagesData[int(len(imagesData)*0.75):]
y2=labels[int(len(labels)*0.75):]

#train the moddel
with tf.Session() as sess:
    cnn=CNN(sess)
    step=0

    for i in range(1000):
        print('Training epoch ({}/{})'.format(i, 1000))
        for startBatch in range(0, len(x1), 32):
            endBatch=startBatch+32
            _, summ=sess.run([cnn.opt, cnn.training], feed_dict={cnn.X: np.expand_dims(x1[startBatch:endBatch], axis=-1),
                                                                cnn.labels: y1[startBatch:endBatch]})
            cnn.file.add_summary(summ, step)
            step+=1
        if i%5==0:
            print('testing..')
            acc_list=[]
            for startBatch in range(0, len(x2), 32):
                endBatch=startBatch+32
                pred=sess.run([cnn.accuracy], feed_dict={cnn.X: np.expand_dims(x2[startBatch:endBatch], axis=-1),
                                                        cnn.labels: y2[startBatch:endBatch]})
                acc_list.append(pred)
            summ=sess.run(cnn.testing, feed_dict={cnn.acc: np.mean(acc_list)})
            cnn.file.add_summary(summ, step)
            print(step)