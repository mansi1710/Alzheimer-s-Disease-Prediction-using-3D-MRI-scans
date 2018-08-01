# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 18:46:10 2018

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
import PIL

train_data_dir = 'AlzheimerDataset/train/'
validation_data_dir = 'AlzheimerDataset/test/'

batch_size = 128
img_height = 160
img_width = 160

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
classifier = Sequential()

classifier.add(Conv2D(32, (4, 4), input_shape = (img_height, img_width, 3), activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(MaxPooling2D(pool_size = (4, 4)))

classifier.add(Conv2D(32, (4, 4), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (5, 5)))


classifier.add(Conv2D(32, (4, 4), activation = 'relu'))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 16, activation = 'relu'))
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'softmax'))

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


classifier.fit_generator(train_generator,
        epochs=100,
        validation_data=validation_generator,
        verbose=1,
        )