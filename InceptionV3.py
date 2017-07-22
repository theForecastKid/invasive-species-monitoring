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
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D
from keras import backend as K
from keras import optimizers

train_set = pd.read_csv('../train_labels/train_labels.csv')
test_set = pd.read_csv('../sample_submission.csv')

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    return img

train_img, test_img = [], []
for img_path in tqdm(train_set['name'].iloc[: ]):
    train_img.append(read_img('../train/' + str(img_path) + '.jpg'))
for img_path in tqdm(test_set['name'].iloc[: ]):
    test_img.append(read_img('../test/' + str(img_path) + '.jpg'))

train_img = np.array(train_img, np.float32) / 255
train_label = np.array(train_set['invasive'].iloc[: ])
test_img = np.array(test_img, np.float32) / 255


base_model = InceptionV3(weights='imagenet', include_top=False)

for layer in base_model.layers[:260]:
   layer.trainable = False
#for layer in base_model.layers[260:]:
#   layer.trainable = True
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
x = Dropout(0.6)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#              loss='categorical_crossentropy')

sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.90, nesterov = True)
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

#earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10, verbose=0, mode='auto')

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='auto'),
            keras.callbacks.ModelCheckpoint('baseline.h5', monitor='val_loss', save_best_only=True, verbose=0)]

model.fit_generator(train_datagen.flow(tr_img, tr_label, batch_size = 32), 
                    epochs=100, steps_per_epoch =1721/32, validation_data = (val_img, val_label), callbacks = callbacks)

yp = model.predict(test_img, batch_size = 32)

sample_submission = pd.read_csv("../sample_submission.csv")

sample_submission['invasive'] = yp

sample_submission.to_csv('baseline.csv', index = False)
