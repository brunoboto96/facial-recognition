import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

#Start
train_data_path = 'images/train/cat'
test_data_path = 'images/test'
all_data_path = 'images/allcat'
img_rows = 68
img_cols = 77
epochs = 200
batch_size = 32
num_of_train_samples = 1500*0.5
num_of_test_samples = 1500*0.5

#Image Generator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   validation_split=0.5,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255,validation_split=0.5)

testimg_datagen = ImageDataGenerator()
testimg_generator = testimg_datagen.flow_from_directory('images/testimg',
                                         target_size=(img_rows, img_cols),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=1) 

print(testimg_generator.class_indices)
train_generator = train_datagen.flow_from_directory(all_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    save_to_dir='images/saved/training',
                                                    subset='training',
                                                    color_mode='grayscale',
                                                    class_mode='categorical')
print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(all_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        save_to_dir='images/saved/validation',
                                                        subset='validation',
                                                        color_mode='grayscale',
                                                        class_mode='categorical')

print(validation_generator.class_indices)

# Build model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), input_shape=(img_rows, img_cols, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(16))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

#Train
model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    shuffle=True,
                    validation_steps=num_of_test_samples // batch_size)

model.save("model/model_test.model")


print("\n\nTESTING********\n")
Y_pred = model.predict_generator(testimg_generator, 1 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

#print(Y_pred)
print('Class is: ',*y_pred)


print("\nTESTING********\n\n")

#Confution Matrix and Classification Report
print( num_of_test_samples // batch_size+1)
print(num_of_test_samples)
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred))
