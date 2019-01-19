---
layout:     post
title:      AlexNet for Dogs Vs Cats (Kaggle)
date:       2019-01-18
summary:    Implementing the AlexNet on Dogs Vs Cats dataset from Kaggle
categories: Deep_Learning AlexNet Kaggle
---

The [previous article](http://www.back-prop.com/deep_learning/lenet/mnist/2019/01/17/AlexNet/) discusses the AlexNet architecture 
and its contribution to the field of Deep Learning. The implementation shown in that article is derived from the book 
[Deep Learning for Computer Vision - Practioner Bundel](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/). This
is an excellent book that not only explains the theory and the math behind the various deep learning techniques, but also, shows 
how to efficiently implement it in Keras.  
<br>

In this article we will see how the AlexNet model can be implemented from scratch on a Dog Vs Cat identification dataset. This dataset is a 
part of a [Kaggle Challenge](https://www.kaggle.com/c/dogs-vs-cats/data). What makes this dataset challenging despit having just two classes 
is that it is too large to fit into memory. Keras does allow batch processing of images by passing it just the image paths but this is limited by the I/O 
speed. The book shows an efficient way of data processing using the HDF5 format. This format although expands the dataset to a much larger 
file, the I/O is much faster and hence the training speeds up as well. A detailed discussion of the conversion of the datset to HDF5 format 
and the preprocessing of the images will be discussed in a lter article. But just as a brief note, the preprocessing involved are the same
as discussed in the previous article. Random patches of 227 x 227 are extracted from the original 256 x 256 image. Their horizontal flipps are 
taken to give a large enough augmented dataset.  
<br>

The Adams optimizer was used with a learning rate of lr=0.5e-3 ( the book uses 1e-3). Since the GPU it was trained on had lesser memory 
a batch size of 64 was used unlike the book where batch size of 128 was recommended. This prompted the reduction in the learning rate 
because without the reduction the same accuracy wasn't achieved. A binary_crossentropy loss function was used since there were only two classes.
The model was run for 75 epochs.  
<br>

The model was able to achieve a validation accuracy of 93% slightly higher than what the original vanila implementation quotes.  
Epoch 70/75  
312/312 [==============================] - 444s 1s/step - loss: 0.4294 - acc: 0.9258 - val_loss: 0.4019 - val_acc: 0.9401  
Epoch 71/75  
312/312 [==============================] - 445s 1s/step - loss: 0.4161 - acc: 0.9261 - val_loss: 0.4531 - val_acc: 0.9134  
Epoch 72/75  
312/312 [==============================] - 443s 1s/step - loss: 0.4047 - acc: 0.9300 - val_loss: 0.4226 - val_acc: 0.9302  
Epoch 73/75  
312/312 [==============================] - 444s 1s/step - loss: 0.4178 - acc: 0.9275 - val_loss: 0.4025 - val_acc: 0.9319  
Epoch 74/75  
312/312 [==============================] - 444s 1s/step - loss: 0.4030 - acc: 0.9297 - val_loss: 0.3906 - val_acc: 0.9351  
Epoch 75/75  
312/312 [==============================] - 444s 1s/step - loss: 0.3994 - acc: 0.9300 - val_loss: 0.4124 - val_acc: 0.9306  
<br>
Loss function and Accuracy summaries
<table>
	<tr> 
		<th> Training Accuracy </th>
		<th> Training Loss </th>
	</tr>
	<tr> 
		<th> <img src='/images/acc_dvc.svg' alt="" width="350"/> </th>
		<th> <img src='/images/loss_dvc.svg' alt="" width="350"/> </th>
	</tr>
  <tr> 
		<th> Validation Accuracy </th>
		<th> Validation Loss </th>
	</tr>
  <tr> 
		<th> <img src='/images/val_acc_dvc.svg' alt="" width="350"/> </th>
		<th> <img src='/images/val_loss_dvc.svg' alt="" width="350"/> </th>
	</tr>
</table>
<br>
