# Facial Recognition Assignment CIS006-2 University of Bedfordshire

We've been given a yale dataset composed of 1500 images to perform deep learning.

## Getting Started

We will need to download the dataset into our system and organize the directories accordingly.
```
├── images
│   ├── allcat
│   │   ├── 02
│   │   ├── 03
│   │   ├── 04
│   │   ├── 05
│   │   ├── 06
│   │   ├── 07
│   │   ├── 08
│   │   ├── 09
│   │   ├── 11
│   │   ├── 12
│   │   ├── 13
│   │   ├── 15
│   │   ├── 16
│   │   ├── 17
│   │   ├── 18
│   │   ├── 20
│   │   ├── 22
│   │   ├── 23
│   │   ├── 24
│   │   ├── 25
│   │   ├── 26
│   │   ├── 27
│   │   ├── 28
│   │   ├── 32
│   │   ├── 33
│   │   ├── 34
│   │   ├── 35
│   │   ├── 37
│   │   ├── 38
│   │   └── 39
│   ├── saved
│   │   ├── training
│   │   └── validation
```


### Prerequisites

Download the dataset:

Dataset: (https://mega.nz/#!cn5XVCrT!esmf4Eo9b1OiA8VKzAAAn-xaAFnhpfU3JewR6wjQik0)


Make directories (Inside your folder)
```
mkdir images
cd images
mkdir saved
cd saved
mkdir training
mkdir validation
cd ..
cd ..
mkdir logs2
mkdir model
```


### Installing

Pip install the libraries.
```
pip install keras
pip install sklearn.metrics
pip install matplotlib
```



## Running the tests

Run main.py
```
python main.py
```

### Break down into end to end tests

These were my tests.


![alt 24](https://i.imgur.com/53MaNwk.png)
![alt 27](https://i.imgur.com/p5H5RNU.png)
![alt 25](https://i.imgur.com/mAsLFl4.png)
![alt 28](https://i.imgur.com/5GhazZD.png)


### CNN Layers

This is my model. I used VGG-like model.
(https://keras.io/getting-started/sequential-model-guide/)


![alt Model](https://i.imgur.com/eDL1cWq.png)


```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_9 (Conv2D)            (None, 66, 75, 32)        320       
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 64, 73, 32)        9248      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 32, 36, 32)        0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 32, 36, 32)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 30, 34, 64)        18496     
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 28, 32, 64)        36928     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 14, 16, 64)        0         
_________________________________________________________________
dropout_8 (Dropout)          (None, 14, 16, 64)        0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 14336)             0         
_________________________________________________________________
dense_5 (Dense)              (None, 256)               3670272   
_________________________________________________________________
dropout_9 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 30)                7710      
=================================================================
Total params: 3,742,974
Trainable params: 3,742,974
Non-trainable params: 0
_________________________________________________________________


```

