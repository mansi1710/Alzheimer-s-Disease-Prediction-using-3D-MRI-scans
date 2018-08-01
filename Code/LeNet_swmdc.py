# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 19:58:23 2018

@author: Lenovo
"""

import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, MaxPooling2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from sklearn.model_selection import train_test_split
import time
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K
import PIL

train_data_dir = 'AlzheimerDataset/train/'
validation_data_dir = 'AlzheimerDataset/test/'

batch_size = 64
img_height = 160
img_width = 160
numClasses=4

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

model= Sequential()
inputShape= (img_height, img_width, 3)

model.add(Conv2D(20, 5, padding= 'same', input_shape= inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, 5, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))

model.add(Dense(numClasses))
model.add(Activation('softmax'))

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


model.fit_generator(train_generator,
        epochs=100,
        validation_data=validation_generator,
        verbose=1,
        )
