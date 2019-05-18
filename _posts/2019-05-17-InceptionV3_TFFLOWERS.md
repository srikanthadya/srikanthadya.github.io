---
layout:     post
title:      Classification of Flowers using Tensorflow 2.0
date:       2019-05-17
summary:    Implementing InceptionV3 Deep Learning Model on TF_FLOWERS dataset on Google Colab platform
categories: Deep_Learning InceptionV3 TF_FLOWERS Feature_Extraction Fine_Tuning Google_Colab
---
In this post we will see how we can use tensorflow 2.0 alpha release and the tf.data dataset to load and classify flowers. **tf.data.Dataset** exposes about 29 preprocessed datasets for developers to easliy load and experiment with test datasets. In this post we will be looking at the **tf_flowers** dataset. We will see the details of what the dataset contains shortly but before that we need to install the tensorflow 2.0 version and the latest tensorflow_datasets (tfds) to be able to access this dataset. Lets do that here. 

Note: This notebook was run on Google Colab with GPU enabled 


```
# Installing tensorflow 2.0 Alpha version
!pip install tensorflow-gpu==2.0.0-alpha0
```

```
# Installing tfds nightly build
!pip install tfds-nightly
```

```
from google.colab import drive
drive.mount('/content/drive/')

%cd /content/drive/'My Drive'/'All_Repos'/deep-learning/InceptionV3
```

    Drive already mounted at /content/drive/; to attempt to forcibly remount, call drive.mount("/content/drive/", force_remount=True).
    /content/drive/My Drive/All_Repos/deep-learning/InceptionV3



```

```

# Exploring tfds 

Each machine learning and deep learning task always requires reading data as its starting point in the pipeline. Since every dataset is different and is from a different source, a lot of time is spent in converting it into a format that the tensorflow expects it to be. In this context, tensorflow has come up with a tf.data.Datasets object that has pre cured datasets that can be readily used in the machine learning pipeline. Datasets available in tfds are

> "abstract_reasoning"
"caltech101"
"cats_vs_dogs"
"celeb_a"
"celeb_a_hq"
"cifar10"
"cifar100"
"cifar10_corrupted"
"clevr"
"coco2014"
"colorectal_histology"
"colorectal_histology_large"
"cycle_gan"
"diabetic_retinopathy_detection"
"dsprites"
"dtd"
"emnist"
"fashion_mnist"
"horses_or_humans"
"image_label_folder"
"imagenet2012"
"imagenet2012_corrupted"
"kmnist"
"lsun"
"mnist"
"omniglot"
"open_images_v4"
"oxford_flowers102"
"oxford_iiit_pet"
"quickdraw_bitmap"
"rock_paper_scissors"
"shapes3d"
"smallnorb"
"sun397"
"svhn_cropped"
"tf_flowers"

The **tf_flowers** dataset is not partitioned into train, test and validation by default. To do that we can use the tfds.Split.TRAIN.subsplit function. This expects a tuple indicating the split ratio like so (8,1,1). This splits the data into 80%, 10% and 10% ratio for train, validation and test sets. 


```
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

print(tf.__version__)

SPLIT_WEIGHTS = (8,1,1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

```

    2.0.0-alpha0


## Loading the TF_FLOWERS dataset

We can use tfds.load() function to load the dataset and split it as well. the load function has the following options. 
```
tfds.load(
    name,
    split=None,
    data_dir=None,
    batch_size=1,
    download=True,
    as_supervised=False,
    with_info=False,
    builder_kwargs=None,
    download_and_prepare_kwargs=None,
    as_dataset_kwargs=None,
    try_gcs=False
)
```
with_info=True gives us the metadata about the dataset

as_supervised=True returns the data and label as a tuple (input, label)


```
(raw_train,raw_val,raw_test),metadata = tfds.load('tf_flowers',
                                                 split=list(splits),
                                                 data_dir="./dataset",
                                                 as_supervised=True,
                                                 with_info=True)
```

# Information about the dataset

From the metadata we can see there are a total of 3670 images. With our 80%, 10% and 10% split, we now have 2950 training images. 


```

num_train, num_val, num_test = (
  metadata.splits['train'].num_examples * weight/10 for weight in SPLIT_WEIGHTS
)

print('Number of training samples {}'.format(num_train))
```

    Number of training samples 2936.0


# Preprocess Images

In the next step we will make all the imaes the same size by resizing them to a 128x128 size and normalizing them by converting all channels to be within 0 and 1. 


```
def resize_normalize(image,label):
  
  image = tf.cast(image,tf.float32)
  
  image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
  
  image = image / 255.0
  
  return image,label


def augment(image, label):
  
  image = tf.image.random_flip_left_right(image)
  
  image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
  
  return image,label
```

## The *map* function

In the above block we defined functions to resize and normalize an image and also augment the training set so that we have random flips of the image and image brightness varied randomly. Since tensorflow builds the graph first and only executes them when the session is run. 

The dataset load object uses the map function to call these preprosessors on every element of the dataset. [ref](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map)

```
map

map(
    map_func,
    num_parallel_calls=None
)

Maps map_func across the elements of this dataset.

This transformation applies map_func to each element of this dataset, and returns a new dataset containing the transformed elements, in the same order as they appeared in the input.

```


```
print(raw_train)
print(raw_val)
print(raw_test)

IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

train = raw_train.map(resize_normalize)
val = raw_val.map(resize_normalize)
test = raw_test.map(resize_normalize)
```

    <_OptionsDataset shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
    <_OptionsDataset shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
    <_OptionsDataset shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>


### Augment


```
train = train.map(augment)
```

## Shuffle

To over come the bias caused by sample sequence, we shuffle the dataset randomly "batch_size" number of samples at a time. **tf.data.Dataset** object has a shuffle() function that lets us achieve this.  

```
shuffle
shuffle(
    buffer_size,
    seed=None,
    reshuffle_each_iteration=None
)

Randomly shuffles the elements of this dataset.

This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer, replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
```

## Batch

With large datasets, not all training data can be held in memory before processing. Hence it becomes important to fetch and process smaller batches of the data at a time so that they can be fit into memory. When the model sees all the samples from **n** batches , the model is said to have trained for an **Epoch**. Again **tf.data.Dataset** object has a batch() function that lets us grab a few samples at a time based on the **batch_size** we use

```
batch
batch(
    batch_size,
    drop_remainder=False
)

Combines consecutive elements of this dataset into batches.

The tensors in the resulting element will have an additional outer dimension, which will be batch_size (or N % batch_size for the last element if batch_size does not divide the number of input elements N evenly and drop_remainder is False). If your program depends on the batches having the same outer dimension, you should set the drop_remainder argument to True to prevent the smaller batch from being produced.

```

## Prefetch

The need for prefetch is best explained by the experts. The explanation from tensorflow documentation is shown below


>### Optimizing Performance

>As new computing devices (such as GPUs and TPUs) make it possible to train neural networks at an increasingly fast rate, the CPU processing is prone to becoming the bottleneck. The tf.data API provides users with building blocks to design input pipelines that effectively utilize the CPU, optimizing each step of the ETL process.

>### Pipelining
To perform a training step, you must first extract and transform the training data and then feed it to a model running on an accelerator. However, in a naive synchronous implementation, while the CPU is preparing the data, the accelerator is sitting idle. Conversely, while the accelerator is training the model, the CPU is sitting idle. The training step time is thus the sum of both CPU pre-processing time and the accelerator training time.

>Pipelining overlaps the preprocessing and model execution of a training step. While the accelerator is performing training step N, the CPU is preparing the data for step N+1. Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract and transform the data.

>Without pipelining, the CPU and the GPU/TPU sit idle much of the time:

>![with_pipeline](https://docs.google.com/uc?export=download&id=1yc9sFN4vDfJceuWN_8F9woMBe8-KbOq5)

>With pipelining, idle time diminishes significantly:

>![with_pipeline](https://docs.google.com/uc?export=download&id=1xyB9Ds9ABcLhX_6ur7m1asYuYCtZxE67)

>The tf.data API provides a software pipelining mechanism through the tf.data.Dataset.prefetch transformation, which can be used to decouple the time when data is produced from the time when data is consumed. In particular, the transformation uses a background thread and an internal buffer to prefetch elements from the input dataset ahead of the time they are requested. The number of elements to prefetch should be equal to (or possibly greater than) the number of batches consumed by a single training step. You could either manually tune this value, or set it to tf.data.experimental.AUTOTUNE which will prompt the tf.data runtime to tune the value dynamically at runtime.






```
SHUFFLE_BUFFER_SIZE = 1024
BATCH_SIZE = 32

train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val = val.batch(BATCH_SIZE)
test = test.batch(BATCH_SIZE)
train = train.prefetch(tf.data.experimental.AUTOTUNE)

```

## Vizualizing data 

the .take() function lets us access to one batch of the train dataset which we can use to visualize the images in the dataset


```
import matplotlib.pyplot as plt
%matplotlib inline

get_label_name = metadata.features['label'].int2str

plt.figure(figsize=(12,12)) 
for btch in train.take(1):
  for i in range(9):
    img,lbl = btch[0][i],btch[1][i]
    plt.subplot(3,3,i+1)
    plt.title(get_label_name(lbl.numpy()))
    plt.imshow(img)
plt.show()
```


![png](InceptionV3_TFFLOWERS_files/InceptionV3_TFFLOWERS_20_0.png)


# Building a Sequential Model using tf.keras

There are two APIs for defining a network a **Sequntial** API and the **Functional** API. 

**Sequential** API lets us define a network layer by layer. The drawback being that it cannot create networks which share layers or have multiple inputs and outputs

**Functional** API lets us define a netwrok in which any layer can connect to any other layer. This adds a lot of flexibility to define complex networks like residual networks and siamese network

In the block below a CNN is defined which contains an Input layer followed by a couple of convolution layer folloewed by maxpooling layers and then finally followed by a couple of dense layers.


```

from tensorflow import keras
keras.__version__
# Creating a simple CNN model in keras using functional API
def create_model():
    img_inputs = keras.Input(shape=IMG_SHAPE)
    conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(img_inputs)
    maxpool_1 = keras.layers.MaxPooling2D((2, 2))(conv_1)
    conv_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_1)
    maxpool_2 = keras.layers.MaxPooling2D((2, 2))(conv_2)
    conv_3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2)
    flatten = keras.layers.Flatten()(conv_3)
    dense_1 = keras.layers.Dense(64, activation='relu')(flatten)
    output = keras.layers.Dense(metadata.features['label'].num_classes, activation='softmax')(dense_1)

    model = keras.Model(inputs=img_inputs, outputs=output)
    
    return model
```

## Visualizing the CNN


```

simple_model = create_model()
simple_model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 128, 128, 3)]     0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 126, 126, 32)      896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 63, 63, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 61, 61, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 30, 30, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 28, 28, 64)        36928     
    _________________________________________________________________
    flatten (Flatten)            (None, 50176)             0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                3211328   
    _________________________________________________________________
    dense_1 (Dense)              (None, 5)                 325       
    =================================================================
    Total params: 3,267,973
    Trainable params: 3,267,973
    Non-trainable params: 0
    _________________________________________________________________



```
keras.utils.plot_model(simple_model, 'flower_model_with_shape_info.png', show_shapes=True)
```




![png](InceptionV3_TFFLOWERS_files/InceptionV3_TFFLOWERS_25_0.png)




```
import datetime, os

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir)
```


```
num_train, num_val, num_test = (
  metadata.splits['train'].num_examples * weight/10 for weight in SPLIT_WEIGHTS
)

steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = round(num_val)//BATCH_SIZE

print('Number of examples in the train set:', num_train)
print('Number of examples in the validation set:', num_val)
print('Number of examples in the test set:', num_test)
```

    Number of examples in the train set: 2936.0
    Number of examples in the validation set: 367.0
    Number of examples in the test set: 367.0



```
def train_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Creating Keras callbacks 
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
    os.makedirs('training_checkpoints/', exist_ok=True)
    early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=20)

    history = model.fit(train.repeat(),
              epochs=20, 
              steps_per_epoch=steps_per_epoch,
              validation_data=val.repeat(),
              validation_steps=validation_steps,
              callbacks=[tensorboard_callback,
                         model_checkpoint_callback,
                         early_stopping_checkpoint])
    
    return history
```


```

history = train_model(simple_model)
```

    Epoch 1/20
    91/91 [==============================] - 15s 162ms/step - loss: 0.7061 - accuracy: 0.7256 - val_loss: 1.8323 - val_accuracy: 0.5881
    Epoch 2/20
    91/91 [==============================] - 14s 150ms/step - loss: 0.6580 - accuracy: 0.7668 - val_loss: 1.8037 - val_accuracy: 0.5938
    Epoch 3/20
    91/91 [==============================] - 14s 149ms/step - loss: 0.5883 - accuracy: 0.7703 - val_loss: 1.8922 - val_accuracy: 0.5938
    Epoch 4/20
    91/91 [==============================] - 13s 148ms/step - loss: 0.5423 - accuracy: 0.7914 - val_loss: 2.1645 - val_accuracy: 0.6136
    Epoch 5/20
    91/91 [==============================] - 14s 151ms/step - loss: 0.4971 - accuracy: 0.8219 - val_loss: 2.2185 - val_accuracy: 0.5568
    Epoch 6/20
    91/91 [==============================] - 14s 151ms/step - loss: 0.4840 - accuracy: 0.8216 - val_loss: 2.5698 - val_accuracy: 0.5824
    Epoch 7/20
    91/91 [==============================] - 14s 151ms/step - loss: 0.4498 - accuracy: 0.8399 - val_loss: 2.4929 - val_accuracy: 0.5852
    Epoch 8/20
    91/91 [==============================] - 14s 151ms/step - loss: 0.4028 - accuracy: 0.8541 - val_loss: 2.7049 - val_accuracy: 0.5795
    Epoch 9/20
    91/91 [==============================] - 14s 155ms/step - loss: 0.4058 - accuracy: 0.8552 - val_loss: 2.6877 - val_accuracy: 0.5625
    Epoch 10/20
    91/91 [==============================] - 14s 156ms/step - loss: 0.3705 - accuracy: 0.8728 - val_loss: 2.8482 - val_accuracy: 0.5994
    Epoch 11/20
    91/91 [==============================] - 14s 150ms/step - loss: 0.3474 - accuracy: 0.8760 - val_loss: 3.4152 - val_accuracy: 0.5455
    Epoch 12/20
    91/91 [==============================] - 14s 154ms/step - loss: 0.3218 - accuracy: 0.8773 - val_loss: 3.4797 - val_accuracy: 0.5852
    Epoch 13/20
    91/91 [==============================] - 14s 150ms/step - loss: 0.2988 - accuracy: 0.8936 - val_loss: 3.6916 - val_accuracy: 0.5739
    Epoch 14/20
    91/91 [==============================] - 14s 157ms/step - loss: 0.2951 - accuracy: 0.8964 - val_loss: 4.1423 - val_accuracy: 0.6023
    Epoch 15/20
    91/91 [==============================] - 14s 155ms/step - loss: 0.2825 - accuracy: 0.8950 - val_loss: 3.8741 - val_accuracy: 0.5881
    Epoch 16/20
    91/91 [==============================] - 14s 152ms/step - loss: 0.2513 - accuracy: 0.9116 - val_loss: 3.9451 - val_accuracy: 0.5966
    Epoch 17/20
    91/91 [==============================] - 14s 153ms/step - loss: 0.2479 - accuracy: 0.9127 - val_loss: 3.9638 - val_accuracy: 0.5625
    Epoch 18/20
    91/91 [==============================] - 14s 154ms/step - loss: 0.2030 - accuracy: 0.9276 - val_loss: 4.6721 - val_accuracy: 0.5653
    Epoch 19/20
    91/91 [==============================] - 14s 150ms/step - loss: 0.2236 - accuracy: 0.9134 - val_loss: 5.4539 - val_accuracy: 0.5597
    Epoch 20/20
    91/91 [==============================] - 14s 156ms/step - loss: 0.1921 - accuracy: 0.9390 - val_loss: 5.9016 - val_accuracy: 0.5795



```
%load_ext tensorboard.notebook
%tensorboard --logdir logs
```


    Output hidden; open in https://colab.research.google.com to view.



```
!kill 6963
```

    /bin/bash: line 0: kill: (6963) - No such process


It is evident from the accuracy curves above that the **model is overfitting**  the training data. The **training accuracy is high but the validation accuracy is very low** . The model is not good enough to generalize to a wide range of images in these classes. This can probably be improved by **adding more data** to the training set and also by using better networks and better regularization techniques like **Dropout and Batchnormalization**.

On the other hand if the validation score was higher than the training score, it would be an indicator that the validation set has ideal images. Regularization techniques would again help in improving the model predictions.

# InceptionV3 model
 
Now let us try to classify the same flowers images using a more complex network the **"InceptionV3"** network. The method of using a network trained on a different but exhaustive dataset for classification of another dataset is called ***Transfer Learning***. In this case we will look at using the InceptionV3 model that was developed by Google on the ImageNet dataset to classify our TF_FLOWERS. The idea behind this being that the network trained on a dataset that consists of 1000 differnt classes would have learnt to identify common features that would be present in our dataset as well. 

A detail discussion on the InceptionV3 network will be in a separate post similar to the one on AlexNet. 

## Loading pretrained InceptionV3 model from keras

**keras.application** object comes with several pretrained models. These models can be used as classifiers, feature extractors or for fine-tuning. 


## From Keras Documentation

>### Available models in keras.application

>Models for image classification with weights trained on ImageNet:
```
Xception
VGG16
VGG19
ResNet, ResNetV2, ResNeXt
InceptionV3
InceptionResNetV2
MobileNet
MobileNetV2
DenseNet
NASNet
```

> **InceptionV3**

>keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
>Inception V3 model, with weights pre-trained on ImageNet.

>This model and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels).

>The default input size for this model is 299x299.

>**Arguments**
>>include_top: whether to include the fully-connected layer at the top of the network.

>>weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).

>>input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.

>>input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (299, 299, 3) (with 'channels_last' data format) or (3, 299, 299) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 75. E.g. (150, 150, 3) would be one valid value.

>>pooling: Optional pooling mode for feature extraction when include_top is False.
None means that the output of the model will be the 4D tensor output of the last convolutional layer.
'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.

>>'max' means that global max pooling will be applied.
classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.

In this first attempt let us use the InceptionV3 network as a feature extractor. For doing this we will load the model except the final dense layer so that we can add a classifier on top of the network. Also since we are using it as a feature extractor, we will not be training (updating the network weights).


```
base_model = keras.applications.InceptionV3(input_shape=IMG_SHAPE,weights='imagenet',include_top=False)

base_model.trainable = False

base_model.summary()
```

    Model: "inception_v3"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 128, 128, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 63, 63, 32)   864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_v1 (BatchNo (None, 63, 63, 32)   96          conv2d[0][0]                     
    __________________________________________________________________________________________________
    activation (Activation)         (None, 63, 63, 32)   0           batch_normalization_v1[0][0]     
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 61, 61, 32)   9216        activation[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_v1_1 (Batch (None, 61, 61, 32)   96          conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 61, 61, 32)   0           batch_normalization_v1_1[0][0]   
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 61, 61, 64)   18432       activation_1[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_v1_2 (Batch (None, 61, 61, 64)   192         conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, 61, 61, 64)   0           batch_normalization_v1_2[0][0]   
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, 30, 30, 64)   0           activation_2[0][0]               
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 30, 30, 80)   5120        max_pooling2d[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_3 (Batch (None, 30, 30, 80)   240         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, 30, 30, 80)   0           batch_normalization_v1_3[0][0]   
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 28, 28, 192)  138240      activation_3[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_v1_4 (Batch (None, 28, 28, 192)  576         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, 28, 28, 192)  0           batch_normalization_v1_4[0][0]   
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 13, 13, 192)  0           activation_4[0][0]               
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 13, 13, 64)   12288       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_v1_8 (Batch (None, 13, 13, 64)   192         conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, 13, 13, 64)   0           batch_normalization_v1_8[0][0]   
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 13, 13, 48)   9216        max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 13, 13, 96)   55296       activation_8[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_v1_6 (Batch (None, 13, 13, 48)   144         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_v1_9 (Batch (None, 13, 13, 96)   288         conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, 13, 13, 48)   0           batch_normalization_v1_6[0][0]   
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, 13, 13, 96)   0           batch_normalization_v1_9[0][0]   
    __________________________________________________________________________________________________
    average_pooling2d (AveragePooli (None, 13, 13, 192)  0           max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 13, 13, 64)   12288       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 13, 13, 64)   76800       activation_6[0][0]               
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 13, 13, 96)   82944       activation_9[0][0]               
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 13, 13, 32)   6144        average_pooling2d[0][0]          
    __________________________________________________________________________________________________
    batch_normalization_v1_5 (Batch (None, 13, 13, 64)   192         conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_v1_7 (Batch (None, 13, 13, 64)   192         conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_v1_10 (Batc (None, 13, 13, 96)   288         conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_11 (Batc (None, 13, 13, 32)   96          conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, 13, 13, 64)   0           batch_normalization_v1_5[0][0]   
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, 13, 13, 64)   0           batch_normalization_v1_7[0][0]   
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, 13, 13, 96)   0           batch_normalization_v1_10[0][0]  
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, 13, 13, 32)   0           batch_normalization_v1_11[0][0]  
    __________________________________________________________________________________________________
    mixed0 (Concatenate)            (None, 13, 13, 256)  0           activation_5[0][0]               
                                                                     activation_7[0][0]               
                                                                     activation_10[0][0]              
                                                                     activation_11[0][0]              
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 13, 13, 64)   16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_15 (Batc (None, 13, 13, 64)   192         conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_15[0][0]  
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 13, 13, 48)   12288       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 13, 13, 96)   55296       activation_15[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_13 (Batc (None, 13, 13, 48)   144         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_16 (Batc (None, 13, 13, 96)   288         conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, 13, 13, 48)   0           batch_normalization_v1_13[0][0]  
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, 13, 13, 96)   0           batch_normalization_v1_16[0][0]  
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, 13, 13, 256)  0           mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 13, 13, 64)   16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 13, 13, 64)   76800       activation_13[0][0]              
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 13, 13, 96)   82944       activation_16[0][0]              
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 13, 13, 64)   16384       average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_v1_12 (Batc (None, 13, 13, 64)   192         conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_14 (Batc (None, 13, 13, 64)   192         conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_17 (Batc (None, 13, 13, 96)   288         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_18 (Batc (None, 13, 13, 64)   192         conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_12[0][0]  
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_14[0][0]  
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, 13, 13, 96)   0           batch_normalization_v1_17[0][0]  
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_18[0][0]  
    __________________________________________________________________________________________________
    mixed1 (Concatenate)            (None, 13, 13, 288)  0           activation_12[0][0]              
                                                                     activation_14[0][0]              
                                                                     activation_17[0][0]              
                                                                     activation_18[0][0]              
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 13, 13, 64)   18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_22 (Batc (None, 13, 13, 64)   192         conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_22[0][0]  
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 13, 13, 48)   13824       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, 13, 13, 96)   55296       activation_22[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_20 (Batc (None, 13, 13, 48)   144         conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_23 (Batc (None, 13, 13, 96)   288         conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, 13, 13, 48)   0           batch_normalization_v1_20[0][0]  
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, 13, 13, 96)   0           batch_normalization_v1_23[0][0]  
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, 13, 13, 288)  0           mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 13, 13, 64)   18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 13, 13, 64)   76800       activation_20[0][0]              
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, 13, 13, 96)   82944       activation_23[0][0]              
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, 13, 13, 64)   18432       average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_v1_19 (Batc (None, 13, 13, 64)   192         conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_21 (Batc (None, 13, 13, 64)   192         conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_24 (Batc (None, 13, 13, 96)   288         conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_25 (Batc (None, 13, 13, 64)   192         conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_19[0][0]  
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_21[0][0]  
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, 13, 13, 96)   0           batch_normalization_v1_24[0][0]  
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_25[0][0]  
    __________________________________________________________________________________________________
    mixed2 (Concatenate)            (None, 13, 13, 288)  0           activation_19[0][0]              
                                                                     activation_21[0][0]              
                                                                     activation_24[0][0]              
                                                                     activation_25[0][0]              
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, 13, 13, 64)   18432       mixed2[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_27 (Batc (None, 13, 13, 64)   192         conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, 13, 13, 64)   0           batch_normalization_v1_27[0][0]  
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, 13, 13, 96)   55296       activation_27[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_28 (Batc (None, 13, 13, 96)   288         conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, 13, 13, 96)   0           batch_normalization_v1_28[0][0]  
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, 6, 6, 384)    995328      mixed2[0][0]                     
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, 6, 6, 96)     82944       activation_28[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_26 (Batc (None, 6, 6, 384)    1152        conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_29 (Batc (None, 6, 6, 96)     288         conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, 6, 6, 384)    0           batch_normalization_v1_26[0][0]  
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, 6, 6, 96)     0           batch_normalization_v1_29[0][0]  
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 6, 6, 288)    0           mixed2[0][0]                     
    __________________________________________________________________________________________________
    mixed3 (Concatenate)            (None, 6, 6, 768)    0           activation_26[0][0]              
                                                                     activation_29[0][0]              
                                                                     max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 6, 6, 128)    98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_34 (Batc (None, 6, 6, 128)    384         conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, 6, 6, 128)    0           batch_normalization_v1_34[0][0]  
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 6, 6, 128)    114688      activation_34[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_35 (Batc (None, 6, 6, 128)    384         conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    activation_35 (Activation)      (None, 6, 6, 128)    0           batch_normalization_v1_35[0][0]  
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 6, 6, 128)    98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 6, 6, 128)    114688      activation_35[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_31 (Batc (None, 6, 6, 128)    384         conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_36 (Batc (None, 6, 6, 128)    384         conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, 6, 6, 128)    0           batch_normalization_v1_31[0][0]  
    __________________________________________________________________________________________________
    activation_36 (Activation)      (None, 6, 6, 128)    0           batch_normalization_v1_36[0][0]  
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 6, 6, 128)    114688      activation_31[0][0]              
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, 6, 6, 128)    114688      activation_36[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_32 (Batc (None, 6, 6, 128)    384         conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_37 (Batc (None, 6, 6, 128)    384         conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, 6, 6, 128)    0           batch_normalization_v1_32[0][0]  
    __________________________________________________________________________________________________
    activation_37 (Activation)      (None, 6, 6, 128)    0           batch_normalization_v1_37[0][0]  
    __________________________________________________________________________________________________
    average_pooling2d_3 (AveragePoo (None, 6, 6, 768)    0           mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 6, 6, 192)    147456      mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 6, 6, 192)    172032      activation_32[0][0]              
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, 6, 6, 192)    172032      activation_37[0][0]              
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, 6, 6, 192)    147456      average_pooling2d_3[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_v1_30 (Batc (None, 6, 6, 192)    576         conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_33 (Batc (None, 6, 6, 192)    576         conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_38 (Batc (None, 6, 6, 192)    576         conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_39 (Batc (None, 6, 6, 192)    576         conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_30[0][0]  
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_33[0][0]  
    __________________________________________________________________________________________________
    activation_38 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_38[0][0]  
    __________________________________________________________________________________________________
    activation_39 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_39[0][0]  
    __________________________________________________________________________________________________
    mixed4 (Concatenate)            (None, 6, 6, 768)    0           activation_30[0][0]              
                                                                     activation_33[0][0]              
                                                                     activation_38[0][0]              
                                                                     activation_39[0][0]              
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, 6, 6, 160)    122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_44 (Batc (None, 6, 6, 160)    480         conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    activation_44 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_44[0][0]  
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, 6, 6, 160)    179200      activation_44[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_45 (Batc (None, 6, 6, 160)    480         conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    activation_45 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_45[0][0]  
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, 6, 6, 160)    122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, 6, 6, 160)    179200      activation_45[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_41 (Batc (None, 6, 6, 160)    480         conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_46 (Batc (None, 6, 6, 160)    480         conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    activation_41 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_41[0][0]  
    __________________________________________________________________________________________________
    activation_46 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_46[0][0]  
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, 6, 6, 160)    179200      activation_41[0][0]              
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, 6, 6, 160)    179200      activation_46[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_42 (Batc (None, 6, 6, 160)    480         conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_47 (Batc (None, 6, 6, 160)    480         conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    activation_42 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_42[0][0]  
    __________________________________________________________________________________________________
    activation_47 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_47[0][0]  
    __________________________________________________________________________________________________
    average_pooling2d_4 (AveragePoo (None, 6, 6, 768)    0           mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, 6, 6, 192)    147456      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, 6, 6, 192)    215040      activation_42[0][0]              
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, 6, 6, 192)    215040      activation_47[0][0]              
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, 6, 6, 192)    147456      average_pooling2d_4[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_v1_40 (Batc (None, 6, 6, 192)    576         conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_43 (Batc (None, 6, 6, 192)    576         conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_48 (Batc (None, 6, 6, 192)    576         conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_49 (Batc (None, 6, 6, 192)    576         conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    activation_40 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_40[0][0]  
    __________________________________________________________________________________________________
    activation_43 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_43[0][0]  
    __________________________________________________________________________________________________
    activation_48 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_48[0][0]  
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_49[0][0]  
    __________________________________________________________________________________________________
    mixed5 (Concatenate)            (None, 6, 6, 768)    0           activation_40[0][0]              
                                                                     activation_43[0][0]              
                                                                     activation_48[0][0]              
                                                                     activation_49[0][0]              
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, 6, 6, 160)    122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_54 (Batc (None, 6, 6, 160)    480         conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    activation_54 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_54[0][0]  
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, 6, 6, 160)    179200      activation_54[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_55 (Batc (None, 6, 6, 160)    480         conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    activation_55 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_55[0][0]  
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, 6, 6, 160)    122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, 6, 6, 160)    179200      activation_55[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_51 (Batc (None, 6, 6, 160)    480         conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_56 (Batc (None, 6, 6, 160)    480         conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    activation_51 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_51[0][0]  
    __________________________________________________________________________________________________
    activation_56 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_56[0][0]  
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, 6, 6, 160)    179200      activation_51[0][0]              
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, 6, 6, 160)    179200      activation_56[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_52 (Batc (None, 6, 6, 160)    480         conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_57 (Batc (None, 6, 6, 160)    480         conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    activation_52 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_52[0][0]  
    __________________________________________________________________________________________________
    activation_57 (Activation)      (None, 6, 6, 160)    0           batch_normalization_v1_57[0][0]  
    __________________________________________________________________________________________________
    average_pooling2d_5 (AveragePoo (None, 6, 6, 768)    0           mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, 6, 6, 192)    147456      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, 6, 6, 192)    215040      activation_52[0][0]              
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, 6, 6, 192)    215040      activation_57[0][0]              
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, 6, 6, 192)    147456      average_pooling2d_5[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_v1_50 (Batc (None, 6, 6, 192)    576         conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_53 (Batc (None, 6, 6, 192)    576         conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_58 (Batc (None, 6, 6, 192)    576         conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_59 (Batc (None, 6, 6, 192)    576         conv2d_59[0][0]                  
    __________________________________________________________________________________________________
    activation_50 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_50[0][0]  
    __________________________________________________________________________________________________
    activation_53 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_53[0][0]  
    __________________________________________________________________________________________________
    activation_58 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_58[0][0]  
    __________________________________________________________________________________________________
    activation_59 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_59[0][0]  
    __________________________________________________________________________________________________
    mixed6 (Concatenate)            (None, 6, 6, 768)    0           activation_50[0][0]              
                                                                     activation_53[0][0]              
                                                                     activation_58[0][0]              
                                                                     activation_59[0][0]              
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, 6, 6, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_64 (Batc (None, 6, 6, 192)    576         conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    activation_64 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_64[0][0]  
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, 6, 6, 192)    258048      activation_64[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_65 (Batc (None, 6, 6, 192)    576         conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    activation_65 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_65[0][0]  
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, 6, 6, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, 6, 6, 192)    258048      activation_65[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_61 (Batc (None, 6, 6, 192)    576         conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_66 (Batc (None, 6, 6, 192)    576         conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    activation_61 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_61[0][0]  
    __________________________________________________________________________________________________
    activation_66 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_66[0][0]  
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, 6, 6, 192)    258048      activation_61[0][0]              
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, 6, 6, 192)    258048      activation_66[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_62 (Batc (None, 6, 6, 192)    576         conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_67 (Batc (None, 6, 6, 192)    576         conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    activation_62 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_62[0][0]  
    __________________________________________________________________________________________________
    activation_67 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_67[0][0]  
    __________________________________________________________________________________________________
    average_pooling2d_6 (AveragePoo (None, 6, 6, 768)    0           mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, 6, 6, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, 6, 6, 192)    258048      activation_62[0][0]              
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, 6, 6, 192)    258048      activation_67[0][0]              
    __________________________________________________________________________________________________
    conv2d_69 (Conv2D)              (None, 6, 6, 192)    147456      average_pooling2d_6[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_v1_60 (Batc (None, 6, 6, 192)    576         conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_63 (Batc (None, 6, 6, 192)    576         conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_68 (Batc (None, 6, 6, 192)    576         conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_69 (Batc (None, 6, 6, 192)    576         conv2d_69[0][0]                  
    __________________________________________________________________________________________________
    activation_60 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_60[0][0]  
    __________________________________________________________________________________________________
    activation_63 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_63[0][0]  
    __________________________________________________________________________________________________
    activation_68 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_68[0][0]  
    __________________________________________________________________________________________________
    activation_69 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_69[0][0]  
    __________________________________________________________________________________________________
    mixed7 (Concatenate)            (None, 6, 6, 768)    0           activation_60[0][0]              
                                                                     activation_63[0][0]              
                                                                     activation_68[0][0]              
                                                                     activation_69[0][0]              
    __________________________________________________________________________________________________
    conv2d_72 (Conv2D)              (None, 6, 6, 192)    147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_72 (Batc (None, 6, 6, 192)    576         conv2d_72[0][0]                  
    __________________________________________________________________________________________________
    activation_72 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_72[0][0]  
    __________________________________________________________________________________________________
    conv2d_73 (Conv2D)              (None, 6, 6, 192)    258048      activation_72[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_73 (Batc (None, 6, 6, 192)    576         conv2d_73[0][0]                  
    __________________________________________________________________________________________________
    activation_73 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_73[0][0]  
    __________________________________________________________________________________________________
    conv2d_70 (Conv2D)              (None, 6, 6, 192)    147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    conv2d_74 (Conv2D)              (None, 6, 6, 192)    258048      activation_73[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_70 (Batc (None, 6, 6, 192)    576         conv2d_70[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_74 (Batc (None, 6, 6, 192)    576         conv2d_74[0][0]                  
    __________________________________________________________________________________________________
    activation_70 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_70[0][0]  
    __________________________________________________________________________________________________
    activation_74 (Activation)      (None, 6, 6, 192)    0           batch_normalization_v1_74[0][0]  
    __________________________________________________________________________________________________
    conv2d_71 (Conv2D)              (None, 2, 2, 320)    552960      activation_70[0][0]              
    __________________________________________________________________________________________________
    conv2d_75 (Conv2D)              (None, 2, 2, 192)    331776      activation_74[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_71 (Batc (None, 2, 2, 320)    960         conv2d_71[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_75 (Batc (None, 2, 2, 192)    576         conv2d_75[0][0]                  
    __________________________________________________________________________________________________
    activation_71 (Activation)      (None, 2, 2, 320)    0           batch_normalization_v1_71[0][0]  
    __________________________________________________________________________________________________
    activation_75 (Activation)      (None, 2, 2, 192)    0           batch_normalization_v1_75[0][0]  
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 2, 2, 768)    0           mixed7[0][0]                     
    __________________________________________________________________________________________________
    mixed8 (Concatenate)            (None, 2, 2, 1280)   0           activation_71[0][0]              
                                                                     activation_75[0][0]              
                                                                     max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    conv2d_80 (Conv2D)              (None, 2, 2, 448)    573440      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_80 (Batc (None, 2, 2, 448)    1344        conv2d_80[0][0]                  
    __________________________________________________________________________________________________
    activation_80 (Activation)      (None, 2, 2, 448)    0           batch_normalization_v1_80[0][0]  
    __________________________________________________________________________________________________
    conv2d_77 (Conv2D)              (None, 2, 2, 384)    491520      mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_81 (Conv2D)              (None, 2, 2, 384)    1548288     activation_80[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_77 (Batc (None, 2, 2, 384)    1152        conv2d_77[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_81 (Batc (None, 2, 2, 384)    1152        conv2d_81[0][0]                  
    __________________________________________________________________________________________________
    activation_77 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_77[0][0]  
    __________________________________________________________________________________________________
    activation_81 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_81[0][0]  
    __________________________________________________________________________________________________
    conv2d_78 (Conv2D)              (None, 2, 2, 384)    442368      activation_77[0][0]              
    __________________________________________________________________________________________________
    conv2d_79 (Conv2D)              (None, 2, 2, 384)    442368      activation_77[0][0]              
    __________________________________________________________________________________________________
    conv2d_82 (Conv2D)              (None, 2, 2, 384)    442368      activation_81[0][0]              
    __________________________________________________________________________________________________
    conv2d_83 (Conv2D)              (None, 2, 2, 384)    442368      activation_81[0][0]              
    __________________________________________________________________________________________________
    average_pooling2d_7 (AveragePoo (None, 2, 2, 1280)   0           mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_76 (Conv2D)              (None, 2, 2, 320)    409600      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_78 (Batc (None, 2, 2, 384)    1152        conv2d_78[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_79 (Batc (None, 2, 2, 384)    1152        conv2d_79[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_82 (Batc (None, 2, 2, 384)    1152        conv2d_82[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_83 (Batc (None, 2, 2, 384)    1152        conv2d_83[0][0]                  
    __________________________________________________________________________________________________
    conv2d_84 (Conv2D)              (None, 2, 2, 192)    245760      average_pooling2d_7[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_v1_76 (Batc (None, 2, 2, 320)    960         conv2d_76[0][0]                  
    __________________________________________________________________________________________________
    activation_78 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_78[0][0]  
    __________________________________________________________________________________________________
    activation_79 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_79[0][0]  
    __________________________________________________________________________________________________
    activation_82 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_82[0][0]  
    __________________________________________________________________________________________________
    activation_83 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_83[0][0]  
    __________________________________________________________________________________________________
    batch_normalization_v1_84 (Batc (None, 2, 2, 192)    576         conv2d_84[0][0]                  
    __________________________________________________________________________________________________
    activation_76 (Activation)      (None, 2, 2, 320)    0           batch_normalization_v1_76[0][0]  
    __________________________________________________________________________________________________
    mixed9_0 (Concatenate)          (None, 2, 2, 768)    0           activation_78[0][0]              
                                                                     activation_79[0][0]              
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 2, 2, 768)    0           activation_82[0][0]              
                                                                     activation_83[0][0]              
    __________________________________________________________________________________________________
    activation_84 (Activation)      (None, 2, 2, 192)    0           batch_normalization_v1_84[0][0]  
    __________________________________________________________________________________________________
    mixed9 (Concatenate)            (None, 2, 2, 2048)   0           activation_76[0][0]              
                                                                     mixed9_0[0][0]                   
                                                                     concatenate[0][0]                
                                                                     activation_84[0][0]              
    __________________________________________________________________________________________________
    conv2d_89 (Conv2D)              (None, 2, 2, 448)    917504      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_89 (Batc (None, 2, 2, 448)    1344        conv2d_89[0][0]                  
    __________________________________________________________________________________________________
    activation_89 (Activation)      (None, 2, 2, 448)    0           batch_normalization_v1_89[0][0]  
    __________________________________________________________________________________________________
    conv2d_86 (Conv2D)              (None, 2, 2, 384)    786432      mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_90 (Conv2D)              (None, 2, 2, 384)    1548288     activation_89[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_v1_86 (Batc (None, 2, 2, 384)    1152        conv2d_86[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_90 (Batc (None, 2, 2, 384)    1152        conv2d_90[0][0]                  
    __________________________________________________________________________________________________
    activation_86 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_86[0][0]  
    __________________________________________________________________________________________________
    activation_90 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_90[0][0]  
    __________________________________________________________________________________________________
    conv2d_87 (Conv2D)              (None, 2, 2, 384)    442368      activation_86[0][0]              
    __________________________________________________________________________________________________
    conv2d_88 (Conv2D)              (None, 2, 2, 384)    442368      activation_86[0][0]              
    __________________________________________________________________________________________________
    conv2d_91 (Conv2D)              (None, 2, 2, 384)    442368      activation_90[0][0]              
    __________________________________________________________________________________________________
    conv2d_92 (Conv2D)              (None, 2, 2, 384)    442368      activation_90[0][0]              
    __________________________________________________________________________________________________
    average_pooling2d_8 (AveragePoo (None, 2, 2, 2048)   0           mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_85 (Conv2D)              (None, 2, 2, 320)    655360      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_v1_87 (Batc (None, 2, 2, 384)    1152        conv2d_87[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_88 (Batc (None, 2, 2, 384)    1152        conv2d_88[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_91 (Batc (None, 2, 2, 384)    1152        conv2d_91[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_v1_92 (Batc (None, 2, 2, 384)    1152        conv2d_92[0][0]                  
    __________________________________________________________________________________________________
    conv2d_93 (Conv2D)              (None, 2, 2, 192)    393216      average_pooling2d_8[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_v1_85 (Batc (None, 2, 2, 320)    960         conv2d_85[0][0]                  
    __________________________________________________________________________________________________
    activation_87 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_87[0][0]  
    __________________________________________________________________________________________________
    activation_88 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_88[0][0]  
    __________________________________________________________________________________________________
    activation_91 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_91[0][0]  
    __________________________________________________________________________________________________
    activation_92 (Activation)      (None, 2, 2, 384)    0           batch_normalization_v1_92[0][0]  
    __________________________________________________________________________________________________
    batch_normalization_v1_93 (Batc (None, 2, 2, 192)    576         conv2d_93[0][0]                  
    __________________________________________________________________________________________________
    activation_85 (Activation)      (None, 2, 2, 320)    0           batch_normalization_v1_85[0][0]  
    __________________________________________________________________________________________________
    mixed9_1 (Concatenate)          (None, 2, 2, 768)    0           activation_87[0][0]              
                                                                     activation_88[0][0]              
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 2, 2, 768)    0           activation_91[0][0]              
                                                                     activation_92[0][0]              
    __________________________________________________________________________________________________
    activation_93 (Activation)      (None, 2, 2, 192)    0           batch_normalization_v1_93[0][0]  
    __________________________________________________________________________________________________
    mixed10 (Concatenate)           (None, 2, 2, 2048)   0           activation_85[0][0]              
                                                                     mixed9_1[0][0]                   
                                                                     concatenate_1[0][0]              
                                                                     activation_93[0][0]              
    ==================================================================================================
    Total params: 21,802,784
    Trainable params: 0
    Non-trainable params: 21,802,784
    __________________________________________________________________________________________________





```
def inceptionV3_classifier(base_model):
  
  model = keras.models.Sequential()
  model.add(base_model)
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(metadata.features['label'].num_classes)) 
  model.add(keras.layers.Activation('softmax'))
  
  model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  return model

inception_model = inceptionV3_classifier(base_model)
```


```

inception_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    inception_v3 (Model)         (None, 2, 2, 2048)        21802784  
    _________________________________________________________________
    flatten (Flatten)            (None, 8192)              0         
    _________________________________________________________________
    dense (Dense)                (None, 5)                 40965     
    _________________________________________________________________
    activation_94 (Activation)   (None, 5)                 0         
    =================================================================
    Total params: 21,843,749
    Trainable params: 40,965
    Non-trainable params: 21,802,784
    _________________________________________________________________



```
import datetime, os
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir)

```

## Defining callbacks and  Training the model 




```

# Creating Keras callbacks 
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
os.makedirs('training_checkpoints/', exist_ok=True)
early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=5)



history = inception_model.fit(train.repeat(),
                              epochs=5,
                              steps_per_epoch = steps_per_epoch,
                              validation_data=val.repeat(), 
                              validation_steps=validation_steps,
                              callbacks=[tensorboard_callback,
                              model_checkpoint_callback,
                              early_stopping_checkpoint])
```

    Epoch 1/5
     1/91 [..............................] - ETA: 20:19 - loss: 2.1121 - accuracy: 0.0938

    W0518 04:10:56.874109 140165189879680 callbacks.py:236] Method (on_train_batch_end) is slow compared to the batch update (0.265347). Check your callbacks.


     2/91 [..............................] - ETA: 10:12 - loss: 2.3433 - accuracy: 0.1094

    W0518 04:10:56.934914 140165189879680 callbacks.py:236] Method (on_train_batch_end) is slow compared to the batch update (0.104219). Check your callbacks.


    91/91 [==============================] - 25s 275ms/step - loss: 1.3427 - accuracy: 0.4825 - val_loss: 0.9771 - val_accuracy: 0.6591
    Epoch 2/5
    91/91 [==============================] - 12s 131ms/step - loss: 1.0589 - accuracy: 0.6268 - val_loss: 0.9059 - val_accuracy: 0.6733
    Epoch 3/5
    91/91 [==============================] - 12s 127ms/step - loss: 0.8954 - accuracy: 0.6726 - val_loss: 0.9417 - val_accuracy: 0.6761
    Epoch 4/5
    91/91 [==============================] - 11s 126ms/step - loss: 0.8835 - accuracy: 0.6781 - val_loss: 1.0322 - val_accuracy: 0.6847
    Epoch 5/5
    91/91 [==============================] - 13s 138ms/step - loss: 0.8406 - accuracy: 0.7020 - val_loss: 1.0506 - val_accuracy: 0.6818



```
import matplotlib.pyplot as plt
%matplotlib inline

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```


![png](InceptionV3_TFFLOWERS_files/InceptionV3_TFFLOWERS_40_0.png)



```
inception_model.save('inception_v3_128_tf_flowes.h5')
```

We were now able to achieve similar training and validation accuracies thus reducing overfitting. The accuracies are still in the 70% range but then the model wasn't trained over a whole lot of epochs. While we can continue the training for a few more epochs, let us explore the other option of fine tuning this model by letting the weights on more layers be trainable. To do that follow the code below

>Note: This should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable. If you add a randomly initialized classifier on top of a pre-trained model and attempt to train all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights from the classifier) and your pre-trained model will just forget everything it has learned.

>Additionally, the reasoning behind fine-tuning the top layers of the pre-trained model rather than all layers of the pre-trained model is the following: in a convnet, the higher up a layer is, the more specialized it is. The first few layers in a convnet learned very simple and generic features, which generalize to almost all types of images. But as you go higher up, the features are increasingly more specific to the dataset that the model was trained on. The goal of fine-tuning is to adapt these specialized features to work with the new dataset.



```
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))


# Fine tune from this layer onwards
fine_tune_at = 249

base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = True
  

# Compile the model using a much-lower training rate.
inception_model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

inception_model.summary()
```

    Number of layers in the base model:  311
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    inception_v3 (Model)         (None, 2, 2, 2048)        21802784  
    _________________________________________________________________
    flatten (Flatten)            (None, 8192)              0         
    _________________________________________________________________
    dense (Dense)                (None, 5)                 40965     
    _________________________________________________________________
    activation_94 (Activation)   (None, 5)                 0         
    =================================================================
    Total params: 21,843,749
    Trainable params: 21,809,317
    Non-trainable params: 34,432
    _________________________________________________________________



```
history_fine = inception_model.fit(train.repeat(), 
                                  steps_per_epoch = steps_per_epoch,
                                  epochs=30,
                                  validation_data=val.repeat(), 
                                  validation_steps=validation_steps,
                                  callbacks=[tensorboard_callback,
                                  model_checkpoint_callback,
                                  early_stopping_checkpoint])
```

    Epoch 1/30
    91/91 [==============================] - 96s 1s/step - loss: 0.6563 - accuracy: 0.7751 - val_loss: 0.9333 - val_accuracy: 0.7841
    Epoch 2/30
    91/91 [==============================] - 20s 216ms/step - loss: 0.3317 - accuracy: 0.8870 - val_loss: 0.6980 - val_accuracy: 0.7983
    Epoch 3/30
    91/91 [==============================] - 20s 215ms/step - loss: 0.1878 - accuracy: 0.9376 - val_loss: 0.6642 - val_accuracy: 0.8494
    Epoch 4/30
    91/91 [==============================] - 19s 206ms/step - loss: 0.0953 - accuracy: 0.9723 - val_loss: 0.8066 - val_accuracy: 0.8153
    Epoch 5/30
    91/91 [==============================] - 20s 218ms/step - loss: 0.0852 - accuracy: 0.9719 - val_loss: 0.7655 - val_accuracy: 0.8494
    Epoch 6/30
    91/91 [==============================] - 20s 215ms/step - loss: 0.0699 - accuracy: 0.9782 - val_loss: 0.8593 - val_accuracy: 0.8210
    Epoch 7/30
    91/91 [==============================] - 20s 216ms/step - loss: 0.0531 - accuracy: 0.9851 - val_loss: 0.7129 - val_accuracy: 0.8693
    Epoch 8/30
    91/91 [==============================] - 19s 205ms/step - loss: 0.0479 - accuracy: 0.9809 - val_loss: 0.5182 - val_accuracy: 0.8864
    Epoch 9/30
    91/91 [==============================] - 19s 206ms/step - loss: 0.0567 - accuracy: 0.9813 - val_loss: 0.7580 - val_accuracy: 0.8438
    Epoch 10/30
    91/91 [==============================] - 21s 231ms/step - loss: 0.0510 - accuracy: 0.9816 - val_loss: 0.6335 - val_accuracy: 0.8864
    Epoch 11/30
    91/91 [==============================] - 20s 221ms/step - loss: 0.0346 - accuracy: 0.9865 - val_loss: 0.8021 - val_accuracy: 0.8580
    Epoch 12/30
    91/91 [==============================] - 19s 205ms/step - loss: 0.0370 - accuracy: 0.9868 - val_loss: 0.8109 - val_accuracy: 0.8693
    Epoch 13/30
    91/91 [==============================] - 19s 206ms/step - loss: 0.0312 - accuracy: 0.9889 - val_loss: 0.7873 - val_accuracy: 0.8608


The model stopped training since the early_stopping callback made it quit owing to the fact that there was not much improvement to the moitored quantity namely accuracy. 

Even with just 13 epochs we see that the training accuracy has now reached 98% and the validation accuracy is at 86%. This is certainly a much better improvement over our previous attempt which was at 70%. But its very evident from the accuracy plots that the model is overfitting our data. This can perhaps be improved some regularization techniques


```
acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```


![png](InceptionV3_TFFLOWERS_files/InceptionV3_TFFLOWERS_46_0.png)



```

```
