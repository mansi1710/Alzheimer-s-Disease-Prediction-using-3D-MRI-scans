# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 12:01:20 2018

@author: Lenovo
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications 
import cv2
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
img_width, img_height = 150, 150 



train_data_dir = 'data_00/train'
validation_data_dir = 'data_00/validation'
nb_train_samples = 5120
nb_validation_samples = 1280
epochs = 100
batch_size = 40 

img_dir= "data_00/train/Alzheimers"
img_dir_2= "data_00/train/nonalzheimers"
img_dir_3= "data_00/validation/Alzheimers"
img_dir_4= "data_00/validation/nonalzheimers"

'''
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    
print(len(data))
print(data[0].shape)
plt.imshow(data[0]);
'''
print("file extracting")
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
train_x1 = np.array([np.array(Image.open(fname).convert("1")) for fname in files])

print("train_x1 extracted")
print(train_x1.shape)

data_path_2 = os.path.join(img_dir_2, '*g')
files_2 = glob.glob(data_path_2)
train_x2=  np.array([np.array(Image.open(fname).convert("1")) for fname in files_2])

print("train_x2 extracted")

data_path_3 = os.path.join(img_dir_3, '*g')
files_3 = glob.glob(data_path_3)
train_x3=  np.array([np.array(Image.open(fname).convert("1")) for fname in files_3])

print("train_x3 extracted")

data_path_4 = os.path.join(img_dir_4, '*g')
files_4 = glob.glob(data_path_4)
train_x4=  np.array([np.array(Image.open(fname).convert("1")) for fname in files_4])

print("train_x4 extracted")

train_y1= np.ones((2560, 1))
train_y2= np.zeros((2560, 1))
train_y3= np.ones((640, 1))
train_y4= np.zeros((640, 1))

print("file extracted")
print("training started")

train_x= np.concatenate((train_x1, train_x2, train_x3, train_x4), axis=0);
train_y= np.concatenate((train_y1, train_y2, train_y3, train_y4), axis=0)

train_x= train_x.reshape(train_x.shape[0], -1)
print(train_x.shape)

train_x= train_x/255.


np.random.seed(7)
#define models
model= Sequential()
model.add(Dense(10000, input_dim=36608, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, validation_split=0.2, epochs=25, batch_size=10)
