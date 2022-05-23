import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

names = ['anger','contempt','disgust','fear','happy','sadness','surprise']
model = keras.models.load_model('model_keras.h5')
img = cv2.imread('istockphoto-175174559-1024x1024.jpg')
img_resize = cv2.resize(img, (48, 48))
img = (np.expand_dims(img_resize,0))
pred = model.predict(img)
print("prediction is ", pred)
classes_x=np.argmax(pred,axis=1)
print("classes", classes_x)
classn= int(classes_x)
print("class is ", names[classn])
