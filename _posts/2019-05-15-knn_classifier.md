---
layout:     post
title:      k-NN for MNIST Classification
date:       2019-05-16
summary:    MNIST hand written digits classification using KNN
categories: Deep_Learning KNN MNIST
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

In the first post we had seen how LeNet was able to achieve a 99% accuracy in classifying the hand written digits of the MNIST dataset. In this post we will see how for a well cured dataset like the MNIST, even simple ML algorithms like KNN will be able to achieve similar classification accuracies. The key here being, "well cured". Each and every image has been heavily pre-processed — including cropping, perfect thresholding, and centered and hence does not accurately represent the real world handwritten digit problem. Never the less it is worth noting that the same classification task can be achieved using a much simpler algorithm. 
<br>
<br>

The k-Nearest Neighbor classifier is by far the most simple machine learning and image classification
algorithm. It doesn’t actually “learn” anything. Instead, this
algorithm relies on the distance between feature vectors 
Simply put, the **k-NN** algorithm classifies unknown data points by finding the most common
class among the **k** closest examples. Each data point in the **k** closest data points casts a vote, and the category with the highest number of votes wins.
<br>
<br>
Predictions are made for a new instance *(x)* by searching through the entire training set for the **k** most similar instances (the neighbors) and summarizing the output variable for those **k** instances. For regression this might be the mean output variable, in classification this might be the mode (or most common) class value.
<br>
<br>
To determine which of the **k** instances in the training dataset are most similar to a new input a distance measure is used. For real-valued input variables, the most popular distance measure is Euclidean distance.
<br>
<br>
Euclidean distance is calculated as the square root of the sum of the squared differences between a new point *(q)* and an existing point *(p)* across all input attributes *i*.
<br>
<br>

$$ d(p,q) = \sqrt{\sum\limits_{i=1}^n (q_i -p_i)^2} $$

### k-NN for Regression
When k-NN is used for regression problems the prediction is based on the mean or the median of the K-most similar instances.

###  k-NN for Classification
When k-NN is used for classification, the output can be calculated as the class with the highest frequency from the K-most similar instances. Each instance in essence votes for their class and the class with the most votes is taken as the prediction.

Class probabilities can be calculated as the normalized frequency of samples that belong to each class in the set of K most similar instances for a new data instance. For example, in a binary classification problem (class is 0 or 1):

$$ p(class=0) = \frac{count(class=0)}{(count(class=0)+count(class=1))} $$
<br>
<br>

## Recognizing handwritten digits using MNIST
<br>
<br>

#### Load the dataset.




```python
import numpy as np
import pandas as pd
import pprint
from sklearn.datasets import load_digits
from IPython.display import display, HTML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

mnist = load_digits()
print(mnist.data.shape)
```

    (1797, 64)
    

#### Split the Data into Train and Test Sets



```python
trainData,testData,trainLabel,testLabel = train_test_split(np.array(mnist.data),mnist.target,test_size=0.25,random_state=42)
```

#### Split the Train Data into Train and Validation Sets

The two hyperparameters in KNN Classifier is the number of neighbors parameter ***k*** and the distance measure. In this case we are assuming Euclidian distance as the distance measure. The tunable parameter is ***k***. To tune the optimal value of ***k*** we will use the validation data set. 


```python
trainData,valData,trainLabel,valLabel = train_test_split(trainData,trainLabel,test_size=0.1,random_state=84)

print("training data points: {}".format(len(trainLabel)))
print("validation data points: {}".format(len(valLabel)))
print("testing data points: {}".format(len(testLabel)))
```

    training data points: 1212
    validation data points: 135
    testing data points: 450
    

We will use KNeighborsClassifier function from sklearn and build multiple models for different values of ***k*** and select the optimal one. 


```python
kVals = np.arange(1,30,2)

for k in kVals:
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData,trainLabel)
    
    # evaluate the model and update the accuracies list
    score = model.score(valData, valLabel)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))

```

    k=1, accuracy=99.26%
    k=3, accuracy=99.26%
    k=5, accuracy=99.26%
    k=7, accuracy=99.26%
    k=9, accuracy=99.26%
    k=11, accuracy=99.26%
    k=13, accuracy=99.26%
    k=15, accuracy=99.26%
    k=17, accuracy=98.52%
    k=19, accuracy=98.52%
    k=21, accuracy=97.78%
    k=23, accuracy=97.04%
    k=25, accuracy=97.78%
    k=27, accuracy=97.04%
    k=29, accuracy=97.04%
    

From the test above it looks like k=1 gives an accuracy of almost 99.3% on this dataset. Let us pick this k value and tain or model. This newely trained model will then be used to predict the class for all the samples in the test dataset to measure the model accuracy. sklear also has a classification_report function that reports the accuracy for every class


```python
model = KNeighborsClassifier(n_neighbors=1)
model.fit(trainData,trainLabel)
predictions = model.predict(testData)

print(classification_report(testLabel,predictions))
```

                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00        43
              1       0.95      1.00      0.97        37
              2       1.00      1.00      1.00        38
              3       0.98      0.98      0.98        46
              4       0.98      0.98      0.98        55
              5       0.98      1.00      0.99        59
              6       1.00      1.00      1.00        45
              7       1.00      0.98      0.99        41
              8       0.97      0.95      0.96        38
              9       0.96      0.94      0.95        48
    
    avg / total       0.98      0.98      0.98       450
    
    

This model also yields an accuracy of 98%  with digits like 0,2,6 and 7 being predicted right 100% of the time.


```python

```
