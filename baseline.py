import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import cv2
import os, gc, sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import model_selection
from sklearn import metrics
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

train_set = pd.read_csv('../train_labels/train_labels.csv')
test_set = pd.read_csv('../sample_submission.csv')

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (180, 180))
    return img

train_img, test_img = [], []
for img_path in tqdm(train_set['name'].iloc[: ]):
    train_img.append(read_img('../train/' + str(img_path) + '.jpg'))
for img_path in tqdm(test_set['name'].iloc[: ]):
    test_img.append(read_img('../test/' + str(img_path) + '.jpg'))

train_img = np.array(train_img, np.float32) / 255
train_label = np.array(train_set['invasive'].iloc[: ])
test_img = np.array(test_img, np.float32) / 255


model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(180, 180, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr = 0.0015, decay = 1e-6, momentum = 0.90, nesterov = True)
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])


tr_img, val_img, tr_label, val_label = train_test_split(train_img, train_label, test_size = 0.15)

train_datagen = ImageDataGenerator(
        #featurewise_center=True,
    #featurewise_std_normalization=True,
    #rescale=1. /255,
    rotation_range= 40,
    zoom_range=[0.8, 1.2],
    width_shift_range= 0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode = 'nearest')

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=2, mode='auto'),
            keras.callbacks.ModelCheckpoint('baseline.h5', monitor='val_loss', save_best_only=True, verbose=0)]

model.fit_generator(train_datagen.flow(tr_img, tr_label, batch_size = 32), 
                    epochs=1000, steps_per_epoch =1721/32, validation_data = (val_img, val_label), callbacks = callbacks)

yp = model.predict(test_img, batch_size = 32)

sample_submission = pd.read_csv("../sample_submission.csv")

sample_submission['invasive'] = yp

sample_submission.to_csv('baseline.csv', index = False)
