import csv
import numpy as np

from random import randint
import random

from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img

from sklearn.model_selection import train_test_split
from skimage import color

import sklearn

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def preprocess_resize(image):
    left = 10
    right = 310
    top = 60
    bottom = 135
    return image[top:bottom, left:right, :]

def preprocess_brightness(image):
    # Preprocess step to augment data with brightness correction
    # input image is HSV so brightness correction is applied to channel V.
    correction_brightness = np.random.uniform(0.2,1)
    image[:,:,2] = image[:,:,2]*correction_brightness
    return image


def preprocess_rgb2hsv(image):
    # Conversion from RGB to HSV
    return color.rgb2hsv(image/255) # Divide by 255 to put in correct range

def generator(samples, batch_size=32):
    # Dynamically generate images and steering angle data to send
    # to the model for training and validation.
    num_samples = len(samples)
    correction = 0.1 # This is a parameter to tune.
    while 1: #Loop forever so generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # For each sample in a batch, augmentation
                # and corrections are performed.

                #Randomly select which camera image to use.
                flip = randint(0,1) # 0 = Don't flip, 1 = flip image
                camera = randint(0,2) # 0 = Center, 1 = Left, 2 = Right
                image_path = batch_sample[camera]
                image = load_img(image_path)
                image = preprocess_resize(img_to_array(image))

                angle = float(batch_sample[3])
                # Here the steering angle associated with the left and right
                # camera images are corrected.
                if camera == 1:
                    angle += correction
                elif camera == 2:
                    angle -= correction

                if flip == 1:
                    # Flips images left to right. This provides additional
                    # data variety to feed the model during training.
                    # Correspondingly, the angle is also flipped.
                    image = np.fliplr(image)
                    angle = -1 * angle

                image = preprocess_rgb2hsv(image)
                image = preprocess_brightness(image)

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def nVidia_model(image_shape):

    model = Sequential()
    # Use Lambda layer to zero-center the inputs.
    model.add(Lambda(lambda x: (x / 1.0 - 0.5), input_shape=image_shape))
    # Convolutional layer with 5x5 kernel filters, stride of 2x2,
    # and outputs feature map with 24 layers. Relu activation is performed.
    model.add(Convolution2D(24,5,5, subsample=(2,2)))
    model.add(Activation('relu'))
    # Convolutional layer with 5x5 kernel filters, stride of 2x2,
    # and outputs feature map with 36 layers. Relu activation is performed.
    model.add(Convolution2D(36,5,5, subsample=(2,2)))
    model.add(Activation('relu'))
    # Convolutional layer with 5x5 kernel filters, stride of 2x2,
    # and outputs feature map with 48 layers. Relu activation is performed.
    model.add(Convolution2D(48,5,5, subsample=(2,2)))
    model.add(Activation('relu'))
    # Convolutional layer with 3x3 kernel filters, stride of 1x1,
    # and outputs feature map with 64 layers. Relu activation is performed.
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    # Convolutional layer with 3x3 kernel filters, stride of 1x1,
    # and outputs feature map with 64 layers. Relu activation is performed.
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))

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
#    model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=4)
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                        validation_data = validation_generator,
                        nb_val_samples = len(validation_samples), nb_epoch=4)
    model.save('v2_model.h5')



## My Training Data
lines = []
with open('drive_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    # Use the last loaded line to get image shape information.
    image = load_img(line[0])
    image = preprocess_resize(img_to_array(image))
    image_shape = image.shape

#Randomly split lines of the CSV file into 80% training and 20% validation.
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


nVidia_model(image_shape)
