import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
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

#train the moddel
