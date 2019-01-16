---
layout:     post
title:      LeNet For MNIST
date:       2019-01-15
summary:    Implementing the LeNet Deep Learning Model on MNIST hand written digits data
categories: Deep_Learning LeNet MNIST
---

MNIST is one of the best studied datasets in computer vision. The goal of the dataset is to correctly identify the handwritten digits from 0-9.'NIST' stands for National Institute of Standards and Technology and 'M' for "modified".
A more detailed description of the dataset can be obtained from [here](http://yann.lecun.com/exdb/mnist/)
<br>
![Sample Image](../images/MnistExamples.png)
<br>
In this post we will look into the implementation of LeNet architecture using Keras and tensorflow and train it on the MNIST dataset for digit recognition.
As we will see, with this simple CNN model, we will be able to achieve a very high accuracy on the dataset.
