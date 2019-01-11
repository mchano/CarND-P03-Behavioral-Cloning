# Behavioral Cloning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

by Mas Chano (Feb 2017 Cohort)

### Project Information
This repository contains my implementation of the Behavioral Cloning project for the Udacity Self-Driving Car Engineer Nanodegree (original Udacity project repo can be found [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3)). 


### Project Goals
* Use a car driving simulator provided by Udacity to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track at least once without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[ex_cap_left]: ./examples/left_2017_04_09_12_01_40_628.jpg "Captured Data - Left Camera"
[ex_cap_center]: ./examples/center_2017_04_09_12_01_40_628.jpg "Captured Data - Center Camera"
[ex_cap_right]: ./examples/right_2017_04_09_12_01_40_628.jpg "Captured Data - Right Camera"
[hist_pre_correction]: ./examples/hist_pre_correction.png "Histogram Pre-Correction"
[hist_post_correction]: ./examples/hist_post_correction.png "Histogram Post-Correction"
[cropped]: ./examples/cropped.png "Cropped"
[t2_beginning]: ./examples/t2_beginning.png "Track 2 - Beginning Section"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Environment

The following environment was used for the project.

PC
 * Intel Core i5-4670K
 * Zotac GeForce 1060 GTX AMP! Edition 6GB (1280 CUDA Cores)
 * Ubuntu 16.04 LTS

Drivers
 * CUDA 8.0
 * cuDNN 5.1
 * NVIDIA 375.39

Python 3.5.1
 * TensorFlow 1.0.1
 * Keras 1.2.1

### Description of Files

My project includes the following files:

| File       | Description                                                  |
| ---        | ---                                                          |
| model.py   | Script used to create and train the model.                   |
| drive.py   | Modified drive.py for driving the car in autonomous mode.    |
| model.h5   | File that contains the trained convolutional neural network. |
| writeup.md | Summary of results.                                          |


The `model.py` file contains the code for training and saving the convolutional neural network. The file contains the pipeline I used for training and validating the model, and it contains comments to explain how the code works. When run, `model.py` outputs the trained network as `model.h5`.

Using the Udacity provided simulator and `drive.py`, the car can be driven autonomously around the track by executing the following:
```sh
python drive.py model.h5
```

### Capturing the Training Data

Training data was captured using the Udacity provided Driving Simulator. The Simulation Program was run using a resolution of 640x480 and quality was set to "Simple" (This setting contains shadows). For steering, a mouse was used as it provided more granular steering angle data than when using the keyboard.

Only track 1 was used to capture training data. Center lane driving was performed going in both directions of the track. No recovery driving data was captured. At least 3 laps around the track were made in each direction.

Each recorded data "point" contains the following information.
* file path of center camera image
* file path of left camera image
* file path of right camera image
* Steering angle
* Throttle
* Break
* Speed

An example "point" of captured data at a particular timestamp can be found below. Captured camera data is 320x160x3 where the channels represent the RGB color space.

| ![alt text][ex_cap_left] | ![alt text][ex_cap_center] | ![alt text][ex_cap_right] |
| --- | --- | --- |
| Left Camera | Center Camera | Right Camera |


| Steering Angle | Throttle | Break | Speed |
| --- | --- | --- | --- |
| 0.0754717	| 1	| 0	| 30.19011 |

Here a positive steering angle represents an effort to turn towards the right.

In total, 12538 data points were captured. Considering there are 3 camera images per data point, a total of 37614 camera images are available to train on. However, the data for the steering angle represents the steering performed from the view of the center camera. Therefore, I adjust the steering angle assigned to the left and right images by a small correction of 0.1 for the left camera and -0.1 for the right camera. The correction is performed in the generator and can be observed in the `generator` function. The correction value of 0.1 and -0.1 was achieved through testing.

A histogram of the 37614 camera images with its corresponding steering angle are displayed below, pre-correction and post_correction.

![alt text][hist_pre_correction] ![alt text][hist_post_correction]

We can see that correcting the steering angle also provides a more balanced steering angle distribution. This will help ensure our model doesn't over train on driving straight (angle=0).

### The Model Architecture

As a starting point, I decided to adapt the model described by NVIDIA in their paper titled, "End to End Learning for Self-Driving Cars". The NVIDIA model consisted of 9 layers in total: a normalization layer, 5 convolutional layers, and 3 fully connected layers.

My adaptation of the NVIDIA model was written using Keras and can be found in the `model.py` file. The basic structure of the model is described in the following sections.

#### Model Input
The NVIDIA model took as input YUV images from the cameras. However, for my model I decided to use the HSV color space instead for the sake of convenience but also because of similarities between HSV and YUV. Both HSV and YUV contain a brightness or luminance component. As we saw in Project 1, a large difference in luminance or brightness values in adjacent pixels typically indicate edges. Passing this information to the model will aid in feature extraction.

The simulator training data are output as RGB images which I convert to HSV using the sci-kit image library (version 0.12.3).

```python
hsv = color.rgb2hsv(rgb)
```

Here, `color.rgb2hsv()` requires an input range of [0,1] and outputs in the same range for all channels.

#### Feature Extraction Layers

The first layer of my model takes the HSV image, which is in shape (160,320,3) and shifts the data from [0,1] to [-0.5,0.5]. This is to zero-center the data.

The next three layers are 2D Convolutional layers with 5x5 filter sizes, strides of 2x2, and depths between 24 to 48. Each layer is activated by a RELU function to introduce nonlinearity.

Next, my model uses two 2D Convolutional layers with 3x3 filters of depth 64. Each layer is again activated by a RELU function.

#### High-Level Reasoning Layers

The output of the last 2D Convolutional layer is then flattened and passed to a sequence of fully connected layers with decreasing number of neurons. These fully connected layers perform the high-level reasoning based on the features extracted by the convolutional layers. The end output is the calculated steering angle.

#### Keras Implementation

The implementation in Keras can be seen below:

```python
model.add(Lambda(lambda x: (x / 1.0 - 0.5), input_shape=image_shape))
model.add(Convolution2D(24,5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
```

As seen above, the model was compiled to use the mean-square-error loss function and Adam optimizer. Because the Adam Optimizer is used, learning rate was not tuned manually.

### Training the Model

In order to train the model, the captured data described previously was split into a training set and a validation set. The `train_test_split` function from `sklearn.model` was used to split the data 80% training and 20% test. Splitting the data in this manner should help reduce overfitting in the model.

```python
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

```

To feed the model with data, a generator was programmed. The generator took images from a given data set in batches of 32. The data is shuffled every epoch.

Additionally, the generator dynamically performed augmentation of the data. For each sample in a batch, a camera angle (center, left, or right) was randomly chosen to represent that sample in a particular epoch. Similarly, some samples were randomly chosen to be flipped horizontally. While the total number of samples fed to the model did not change, the spread and variety of samples were artificially increased. This helps prevent overfitting and creates a more robust model.

Finally, the generator converted the RGB image to the HSV color space and fed the data to the model.

In the end, only about 4 epochs were needed to train the model. Anymore and validation loss would not decrease substantially or oscillations in validation loss would be observed.

A file, `model.h5`, is output at the end and is used to autonomously drive the simulated vehicle in the Simulation Program.

### Track 1 Autonomous Mode
Using the `model.h5` generated through training, it is connected to the Simulation program by executing the following:

```sh
python drive.py model.h5
```

The `drive.py` file feeds center camera images from the Simulation program to the model one at a time. The model outputs a steering angle which is passed back to the Simulation program through `drive.py`.

In the end, the model successfully navigated track 1 safely. This can be observed in the MP4 file submitted with this project.

### Track 2 Autonomous Mode (Optional)

Unfortunately, in its current state, the model failed to navigate track 2 and hence meant the model failed to generalize beyond track 1. The model never saw track 2 during training.

In an attempt to generalize the model, I added Dropout layers with drop probability ranging from 0.3 to 0.5 after my fully connected layers.



```python
# The output of the Convolutional layer is flattened before
# passing to the fully connected layers.
model.add(Flatten())
# Fully connected layer with 1164 neurons, followed by Dropout
model.add(Dense(1164))
model.add(Dropout(0.5))
# Fully connected layer with 100 neurons, followed by Dropout
model.add(Dense(100))
model.add(Dropout(0.4))
# Fully connected layer with 50 neurons, followed by Dropout
model.add(Dense(50))
model.add(Dropout(0.3))
# Fully connected layer with 10 neurons, followed by Dropout
model.add(Dense(10))
model.add(Dropout(0.2))
# Fully connected layer outputs the steering angle.
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
```

I also added a few preprocessing steps to my training program such as cropping the images to focus only on the road and ignore scenery. The resulting image of a crop is shown below.

![alt text][cropped]

The code used to crop the image is a simple array manipulation.

```python
def preprocess_resize(image):
    left = 10
    right = 310
    top = 60
    bottom = 135
    return image[top:bottom, left:right, :]
```

Additionally, due to the sharper curves and steeper roads, I slowed down the car in autonomous mode to 9MPH (through `drive.py`). For track 1, I could get the model to run at 30MPH autonomously. Unfortunately, slowing down the speed did not help for track 2.

The car would always get stuck at the beginning, driving towards the center, most probably because it interpreted the road to be wider than it is due to the opposing road running parallel to the road of interest.

![alt text][t2_beginning]  

Manually avoiding the initial section allowed the car to run autonomously for a short while until encountering heavily shadowed or turns much sharper than those observed in track 1.

Based on these observations, I believe it is possible to augment the data from track 1 in such a way that the model can be trained to handle terrain never seen before but it would take creative manipulation of the images.

After testing the models generated by my initial model and the new adjusted model here, I found out that the generalization strategies employed here helped for track 1 as well. The trainings were more consistently producing working models and the models seemed to handle slightly better on the tracks. These changes have been incorporated into `model.py` and `drive.py`. The resulting `model.h5` is also a result of these changes.

Although I wish I could have gotten track 2 to work with just track 1 data, in the interest of time, I will re-examine this project at a later date.
