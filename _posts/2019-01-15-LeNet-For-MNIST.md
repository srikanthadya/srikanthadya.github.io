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
