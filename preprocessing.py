from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

'''
The images are labeled yaleBnumber where number is unique for each person.
So, yaleB02 is a person while yaleB05 is another person.

The program is going to iterate through the image files.
Every image is opened and stored as an array. Then, the image
is associated with its correspondent label.

How the image is associated with the label?
The labels are numbers between 0 and 29.

Let's take an image stored which is called yaleB02,
the programs check if another image of person 02 has been already
saved. If so, we are going to use the same label associated the first time we met number 02.
Otherwise, if it is a new person, a new label is associated with it.

Once the images are all stored and every image has been associated with the correspondt label, 
the data are split into training set and validation set. The training set is going to be used to 
train the CNN, while the validation set is going to be used to test the CNN and see if it actually learning
and not trying to memorize things
'''

def preprocessing(folder):
    persons=[]
    labels=[]
    imagesData=[]
    #Iterate through the images file
    for img in os.listdir(folder):
        #open the image into the program as an array
        myImg=Image.open(folder+'/'+img)
        imagesData.append(np.asarray(myImg))
        #retrieve the person number img[5:7]. so yaleB02[5:7]=02 
        if(img[5:7] in persons):
            #if it has already been added, just associate the image with the label
            labels.append(persons.index(img[5:7]))
        else:
            #if not, create the label for the new person
            persons.append(img[5:7])
            labels.append(persons.index(img[5:7]))
    
    #shuffle the data and split in train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(imagesData), np.asarray(labels), test_size=0.3, random_state=42)

    return np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1), y_train, y_test

def dataAugmentation(images):
    #apply between 0 to 3 data augmentation techniques
    seq = iaa.SomeOf((0,1), [
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
    ], random_order=True) # apply augmenters in random order

    return seq.augment_images(images)