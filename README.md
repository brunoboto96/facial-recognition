# Bio-metric Facial Recognition Algorithm

## Introduction
This project is going to develop a Biometric Face Recognition (BFR) problem using machine learning solutions. 

The task requires to create the most efficient algorithm that is able to recognize 30 different people using a limited amount of data. More precisely, the dataset consists of 50 images for each person, for a total of 1500 images. 
The images are taken in slightly different positions, background and lighting conditions.

The current state of art algorithms in facial recognition involve the use of Deep Learning.
Convolutional Neural Networks or in short CNN, allows to extract features from images. So, Deep Face use CNN as well as different other methods to achieve state of the art performance in face recognition.

## Architecture
![alt text](https://github.com/R-Stefano/facial-recognition/blob/stefano/report_images/networkArchitecture.png)
## Results
In order to explore how the model performs and its biases, we used a confusion matrix to draw the relation between predictions and ground-truth predictions among all the test dataset. The matrix has been enhanced with colors in order to simplify the evaluation process to human eyes.
![alt text](https://github.com/R-Stefano/facial-recognition/blob/stefano/report_images/confusionMatrix.png)
As reported in the image above, the algorithm is able to recognize with an accuracy of 100% the 56% of all the persons. 
Notably, the worst performance is achieved when the model has to classify person 23 and person 25. The network misclassify them with person 25 and person 19 respectively. 
![alt text](https://github.com/R-Stefano/facial-recognition/blob/stefano/report_images/wrongPredictions.png)

