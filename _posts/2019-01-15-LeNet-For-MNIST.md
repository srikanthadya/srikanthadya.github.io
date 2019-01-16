---
layout:     post
title:      LeNet For MNIST
date:       2019-01-15
summary:    Implementing the LeNet Deep Learning Model on MNIST hand written digits data
categories: Deep_Learning LeNet MNIST
---

MNIST is one of the best studied datasets in computer vision. The goal of the dataset is to correctly identify the handwritten digits from 0-9.'NIST' stands for National Institute of Standards and Technology and 'M' for "modified". The dataset is modified to remove any image preprocessing burden so that the focus is solely on machine learning.  
A more detailed description of the dataset can be obtained from [here](http://yann.lecun.com/exdb/mnist/)
<br>
<img src='/images/MnistExamples.png' alt=""/>
<br>
In this post we will look into the implementation of LeNet architecture using Keras and tensorflow and train it on the MNIST dataset for digit recognition.
As we will see, with this simple CNN model, we will be able to achieve a very high accuracy on the dataset.
<br><br>
The Architecture is summarized in the table below

|  Layer Type | Output Size  |  Filter/Stride Size | 
|---|---|---|
| Input Layer  | 28 x 28 x 1  |   |
| Conv2D |  28 x 28 x 20 |  5 x 5  |
| Activation |28 x 28 x 20 |   |
| Pool | 14 x 14 x 20  | 2 x 2  |
| Conv2D | 14 x 14 x 50  | 5 x 5 |
| Activation | 14 x 14 x 50  |   |
| Pool  | 7 x 7 x 50  | 2 x 2  |
| FC  | 500  |   |
| Activation | 500  |   |
| FC | 10  |   |
| Softmax  | 10  |   |

<br><br>

The implementation differs from the original architecture in that the activations used here are ReLU instead of tanh. It is a common practice to increase the Conv2D filters per layer as the spatial input dimensions decrease. 
<br><br>
The code block showing the Keras implementation of this architecture is shown below.
```python
'''
# Created by Srikanth Adya at 1/15/2019

Feature: Cofiguration variables set here
# Build LeNet CONVNET

'''

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # CONV >> RELU >> POOL
        model.add(Conv2D(20, (5,5), padding='same',
                         input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        # CONV >> RELU >> POOL
        model.add(Conv2D(50, (5,5), padding='same',
                         input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        # FC >> DENSE
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # Softmax Classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

```
It is important to scale the input images to the range [0,1]. This ensures faster convergence. The scaling is done by dividing every pixel by 255.0 . The labels need to be one-hot encoded since our output layer is going to be a softmax layer predicting 10 classes. This one-hot encoding takes a class value as an input and converts it into a binary vector with all zeros except for the index of the class. For example, if the class label is 3, the one-hot encoded vector will be [0, 0, 0, 1, 0, 0, 0 , 0, 0, 9]. The 4th index is '1' since the classes are from 0-9. This is achieved in keras using the "to_categorical()" function in "keras.utils.np_utils" .

<br> <br>
The model is trained for 20 epochs with a batch-size of 128 with SGD "mini-batch gradient decent" optimizer and a lerning rate of 0.01.

<br>
<img src='/images/graph_run=.png' alt=""/>
<br>
<img src='/images/acc.svg' alt=""/>
<br>
<img src='/images/loss.svg' alt=""/>
<br>

